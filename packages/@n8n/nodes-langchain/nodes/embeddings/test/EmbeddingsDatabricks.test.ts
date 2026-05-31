import { OpenAIEmbeddings } from '@langchain/openai';
import { createMockExecuteFunction } from 'n8n-nodes-base/test/nodes/Helpers';
import type { INode, ISupplyDataFunctions } from 'n8n-workflow';
import { lookup } from 'node:dns/promises';
import type { Mock, Mocked } from 'vitest';

import { EmbeddingsDatabricks } from '../EmbeddingsDatabricks/EmbeddingsDatabricks.node';

vi.mock('@langchain/openai');

class MockProxyAgent {}

vi.mock('@n8n/ai-utilities', async () => {
	const actual = await vi.importActual('@n8n/ai-utilities');
	return {
		...actual,
		logWrapper: vi.fn().mockImplementation((value: unknown) => value),
		getProxyAgent: vi.fn().mockImplementation(() => new MockProxyAgent()),
		getConnectionHintNoticeField: vi.fn().mockReturnValue({}),
	};
});

vi.mock('node:dns/promises', () => ({
	lookup: vi.fn(),
}));

const MockedOpenAIEmbeddings = vi.mocked(OpenAIEmbeddings);
const mockedLookup = vi.mocked(lookup) as unknown as Mock;

describe('EmbeddingsDatabricks', () => {
	let node: EmbeddingsDatabricks;

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
		const node = { ...mockNode, ...nodeOverrides };
		const mockContext = createMockExecuteFunction<ISupplyDataFunctions>(
			{},
			node,
		) as Mocked<ISupplyDataFunctions>;

		mockContext.getCredentials = vi.fn().mockResolvedValue(databricksCredentials);
		mockContext.getNode = vi.fn().mockReturnValue(node);
		// @ts-expect-error - Mocking
		mockContext.getNodeParameter = vi.fn();
		mockContext.logger = {
			debug: vi.fn(),
			info: vi.fn(),
			warn: vi.fn(),
			error: vi.fn(),
		};
		return mockContext;
	};

	beforeEach(() => {
		node = new EmbeddingsDatabricks();
		vi.clearAllMocks();
		// Default: hostnames resolve to a public, unicast address.
		mockedLookup.mockResolvedValue([{ address: '52.1.2.3', family: 4 }]);
	});

	describe('description', () => {
		it('should have correct node metadata', () => {
			expect(node.description.name).toBe('embeddingsDatabricks');
			expect(node.description.displayName).toBe('Embeddings Databricks');
			expect(node.description.credentials).toEqual([{ name: 'databricksApi', required: true }]);
			expect(node.description.outputs).toEqual(['ai_embedding']);
		});

		it('should load model options via the getModels method', () => {
			const modelProp = node.description.properties.find((p) => p.name === 'model');
			expect(modelProp?.type).toBe('options');
			expect(modelProp?.typeOptions?.loadOptionsMethod).toBe('getModels');
		});
	});

	describe('supplyData', () => {
		it('should create OpenAIEmbeddings pointed at the Databricks serving endpoints base', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'databricks-gte-large-en';
				if (paramName === 'options') return {};
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedOpenAIEmbeddings).toHaveBeenCalledWith(
				expect.objectContaining({
					apiKey: 'dapi-test-token-123',
					model: 'databricks-gte-large-en',
					configuration: expect.objectContaining({
						baseURL: 'https://dbc-test.cloud.databricks.com/serving-endpoints',
					}),
				}),
			);
		});

		it('should strip trailing slash from host', async () => {
			const ctx = setupMockContext();
			ctx.getCredentials = vi.fn().mockResolvedValue({
				host: 'https://dbc-test.cloud.databricks.com/',
				token: 'dapi-test',
			});
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-endpoint';
				if (paramName === 'options') return {};
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedOpenAIEmbeddings).toHaveBeenCalledWith(
				expect.objectContaining({
					configuration: expect.objectContaining({
						baseURL: 'https://dbc-test.cloud.databricks.com/serving-endpoints',
					}),
				}),
			);
		});

		it('should pass batchSize and stripNewLines options through', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return { batchSize: 100, stripNewLines: false };
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedOpenAIEmbeddings).toHaveBeenCalledWith(
				expect.objectContaining({
					batchSize: 100,
					stripNewLines: false,
				}),
			);
		});

		it('should return the embeddings instance as response', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return {};
				return undefined;
			});

			const result = await node.supplyData.call(ctx, 0);

			expect(result.response).toBeInstanceOf(OpenAIEmbeddings);
		});

		it('should reject a host that resolves to a private address', async () => {
			mockedLookup.mockResolvedValue([{ address: '169.254.169.254', family: 4 }]);
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return {};
				return undefined;
			});

			await expect(node.supplyData.call(ctx, 0)).rejects.toThrow(
				'must be a public Databricks workspace URL',
			);
			expect(MockedOpenAIEmbeddings).not.toHaveBeenCalled();
		});
	});
});
