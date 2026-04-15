/* eslint-disable n8n-nodes-base/node-filename-against-convention */
import { createMockExecuteFunction } from 'n8n-nodes-base/test/nodes/Helpers';
import type { INode, ISupplyDataFunctions } from 'n8n-workflow';

jest.mock('@n8n/ai-utilities', () => {
	const actual = jest.requireActual('@n8n/ai-utilities');
	return {
		...actual,
		logWrapper: jest.fn().mockImplementation((obj) => obj),
		getConnectionHintNoticeField: jest.fn().mockReturnValue({}),
	};
});

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

import { EmbeddingsDatabricks } from '../EmbeddingsDatabricks/EmbeddingsDatabricks.node';

describe('EmbeddingsDatabricks', () => {
	let node: EmbeddingsDatabricks;
	let mockContext: jest.Mocked<ISupplyDataFunctions>;

	const mockNode: INode = {
		id: '1',
		name: 'Embeddings Databricks',
		typeVersion: 1,
		type: '@n8n/n8n-nodes-langchain.embeddingsDatabricks',
		position: [0, 0],
		parameters: {},
	};

	const databricksCredentials = {
		host: 'https://dbc-test.cloud.databricks.com',
		token: 'dapi-test-token-123',
	};

	const setupMockContext = (nodeOverrides: Partial<INode> = {}) => {
		const nodeConfig = { ...mockNode, ...nodeOverrides };
		mockContext = createMockExecuteFunction<ISupplyDataFunctions>(
			{},
			nodeConfig,
		) as jest.Mocked<ISupplyDataFunctions>;

		mockContext.getCredentials = jest.fn().mockResolvedValue(databricksCredentials);
		mockContext.getNode = jest.fn().mockReturnValue(nodeConfig);
		mockContext.getNodeParameter = jest.fn();
		mockContext.logger = {
			debug: jest.fn(),
			info: jest.fn(),
			warn: jest.fn(),
			error: jest.fn(),
		};
		return mockContext;
	};

	beforeEach(() => {
		node = new EmbeddingsDatabricks();
		jest.clearAllMocks();
	});

	describe('description', () => {
		it('should have correct node metadata', () => {
			expect(node.description.name).toBe('embeddingsDatabricks');
			expect(node.description.displayName).toBe('Embeddings Databricks');
			expect(node.description.credentials).toEqual([{ name: 'databricksApi', required: true }]);
		});
	});

	describe('supplyData', () => {
		it('should return an embeddings instance with logWrapper', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = jest.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'databricks-gte-large-en';
				return undefined;
			});

			const result = await node.supplyData.call(ctx, 0);

			expect(result.response).toBeDefined();
			expect(ctx.getCredentials).toHaveBeenCalledWith('databricksApi');
		});

		it('should strip trailing slash from host', async () => {
			const ctx = setupMockContext();
			ctx.getCredentials = jest.fn().mockResolvedValue({
				host: 'https://dbc-test.cloud.databricks.com/',
				token: 'dapi-test-token',
			});
			ctx.getNodeParameter = jest.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-endpoint';
				return undefined;
			});

			const result = await node.supplyData.call(ctx, 0);
			const embeddings = result.response as any;

			// Verify the host doesn't have a trailing slash
			expect(embeddings.host).toBe('https://dbc-test.cloud.databricks.com');
		});
	});

	describe('DatabricksEmbeddings class', () => {
		it('should call the correct Databricks API endpoint for embedDocuments', async () => {
			mockFetch.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					data: [{ embedding: [0.1, 0.2, 0.3] }, { embedding: [0.4, 0.5, 0.6] }],
				}),
			});

			const ctx = setupMockContext();
			ctx.getNodeParameter = jest.fn().mockReturnValue('test-embedding-endpoint');

			const { response: embeddings } = await node.supplyData.call(ctx, 0);

			const result = await (embeddings as any).embedDocuments(['hello', 'world']);

			expect(mockFetch).toHaveBeenCalledWith(
				'https://dbc-test.cloud.databricks.com/serving-endpoints/test-embedding-endpoint/invocations',
				expect.objectContaining({
					method: 'POST',
					headers: expect.objectContaining({
						Authorization: 'Bearer dapi-test-token-123',
						'Content-Type': 'application/json',
					}),
					body: JSON.stringify({ input: ['hello', 'world'] }),
				}),
			);
			expect(result).toEqual([
				[0.1, 0.2, 0.3],
				[0.4, 0.5, 0.6],
			]);
		});

		it('should handle legacy predictions response format', async () => {
			mockFetch.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					predictions: [[0.7, 0.8, 0.9]],
				}),
			});

			const ctx = setupMockContext();
			ctx.getNodeParameter = jest.fn().mockReturnValue('legacy-endpoint');

			const { response: embeddings } = await node.supplyData.call(ctx, 0);
			const result = await (embeddings as any).embedDocuments(['test']);

			expect(result).toEqual([[0.7, 0.8, 0.9]]);
		});

		it('should throw on unexpected response format', async () => {
			mockFetch.mockResolvedValueOnce({
				ok: true,
				json: async () => ({ unexpected: 'format' }),
			});

			const ctx = setupMockContext();
			ctx.getNodeParameter = jest.fn().mockReturnValue('bad-endpoint');

			const { response: embeddings } = await node.supplyData.call(ctx, 0);

			await expect((embeddings as any).embedDocuments(['test'])).rejects.toThrow(
				'Unexpected Databricks embeddings API response format',
			);
		});

		it('should throw on API error response', async () => {
			mockFetch.mockResolvedValueOnce({
				ok: false,
				status: 401,
				text: async () => 'Unauthorized',
			});

			const ctx = setupMockContext();
			ctx.getNodeParameter = jest.fn().mockReturnValue('test-endpoint');

			const { response: embeddings } = await node.supplyData.call(ctx, 0);

			await expect((embeddings as any).embedDocuments(['test'])).rejects.toThrow(
				'Databricks embeddings API error (401): Unauthorized',
			);
		});

		it('should embed a single query via embedQuery', async () => {
			mockFetch.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					data: [{ embedding: [0.1, 0.2, 0.3] }],
				}),
			});

			const ctx = setupMockContext();
			ctx.getNodeParameter = jest.fn().mockReturnValue('test-endpoint');

			const { response: embeddings } = await node.supplyData.call(ctx, 0);
			const result = await (embeddings as any).embedQuery('single query');

			expect(result).toEqual([0.1, 0.2, 0.3]);
			expect(mockFetch).toHaveBeenCalledWith(
				expect.any(String),
				expect.objectContaining({
					body: JSON.stringify({ input: ['single query'] }),
				}),
			);
		});
	});
});
