import { mock } from 'jest-mock-extended';
import type { ISupplyDataFunctions } from 'n8n-workflow';

import { DatabricksVectorStore } from './DatabricksVectorStore';

// Mock the createVectorStoreNode factory
jest.mock('@n8n/ai-utilities', () => ({
	metadataFilterField: {},
	getMetadataFiltersValues: jest.fn(),
	logAiEvent: jest.fn(),
	N8nBinaryLoader: class {},
	N8nJsonLoader: class {},
	logWrapper: (fn: unknown) => fn,
	createVectorStoreNode: (config: {
		getVectorStoreClient: (...args: unknown[]) => unknown;
		populateVectorStore: (...args: unknown[]) => unknown;
	}) =>
		class BaseNode {
			async getVectorStoreClient(...args: unknown[]) {
				return config.getVectorStoreClient.apply(config, args);
			}

			async populateVectorStore(...args: unknown[]) {
				return config.populateVectorStore.apply(config, args);
			}
		},
}));

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

import * as DatabricksNode from './VectorStoreDatabricks.node';

describe('VectorStoreDatabricks.node', () => {
	const helpers = mock<ISupplyDataFunctions['helpers']>();
	const dataFunctions = mock<ISupplyDataFunctions>({ helpers });
	dataFunctions.logger = {
		info: jest.fn(),
		debug: jest.fn(),
		error: jest.fn(),
		warn: jest.fn(),
		verbose: jest.fn(),
	} as unknown as ISupplyDataFunctions['logger'];

	const databricksCredentials = {
		host: 'https://dbc-test.cloud.databricks.com',
		token: 'dapi-test-token-123',
	};

	beforeEach(() => {
		jest.clearAllMocks();
	});

	describe('getVectorStoreClient', () => {
		it('should create vector store client with correct config', async () => {
			const mockEmbeddings = {};

			const context = {
				getCredentials: jest.fn().mockResolvedValue(databricksCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						indexName: 'catalog.schema.my_index',
						textColumn: 'content',
						metadataColumns: 'author, category',
						options: { scoreThreshold: 0.7 },
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreDatabricks' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new DatabricksNode.VectorStoreDatabricks();
			const vectorStore = (await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			)) as DatabricksVectorStore;

			expect(vectorStore).toBeInstanceOf(DatabricksVectorStore);
			expect((vectorStore as any).config).toEqual({
				workspaceUrl: 'https://dbc-test.cloud.databricks.com',
				token: 'dapi-test-token-123',
				indexName: 'catalog.schema.my_index',
				textColumn: 'content',
				metadataColumns: ['author', 'category'],
				scoreThreshold: 0.7,
			});
		});

		it('should strip trailing slash from host', async () => {
			const context = {
				getCredentials: jest.fn().mockResolvedValue({
					host: 'https://dbc-test.cloud.databricks.com/',
					token: 'dapi-test',
				}),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						indexName: 'test.index',
						textColumn: 'text',
						metadataColumns: '',
						options: {},
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreDatabricks' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new DatabricksNode.VectorStoreDatabricks();
			const vectorStore = (await (node as any).getVectorStoreClient(
				context,
				undefined,
				{},
				0,
			)) as DatabricksVectorStore;

			expect((vectorStore as any).config.workspaceUrl).toBe(
				'https://dbc-test.cloud.databricks.com',
			);
		});

		it('should handle empty metadata columns', async () => {
			const context = {
				getCredentials: jest.fn().mockResolvedValue(databricksCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						indexName: 'test.index',
						textColumn: 'text',
						metadataColumns: '',
						options: {},
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreDatabricks' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new DatabricksNode.VectorStoreDatabricks();
			const vectorStore = (await (node as any).getVectorStoreClient(
				context,
				undefined,
				{},
				0,
			)) as DatabricksVectorStore;

			expect((vectorStore as any).config.metadataColumns).toEqual([]);
		});
	});

	describe('populateVectorStore', () => {
		it('should call fromDocuments with correct config', async () => {
			const mockEmbeddings = {
				embedDocuments: jest.fn().mockResolvedValue([
					[0.1, 0.2],
					[0.3, 0.4],
				]),
			};
			const mockDocuments = [
				{ pageContent: 'doc 1', metadata: { id: 'a' } },
				{ pageContent: 'doc 2', metadata: { id: 'b' } },
			];

			mockFetch.mockResolvedValue({ ok: true, json: async () => ({}) });

			const context = {
				getCredentials: jest.fn().mockResolvedValue(databricksCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						indexName: 'catalog.schema.my_index',
						textColumn: 'text',
						metadataColumns: '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreDatabricks' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new DatabricksNode.VectorStoreDatabricks();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			expect(mockFetch).toHaveBeenCalledWith(
				'https://dbc-test.cloud.databricks.com/api/2.0/vector-search/indexes/catalog.schema.my_index/upsert-data',
				expect.objectContaining({
					method: 'POST',
					headers: expect.objectContaining({
						Authorization: 'Bearer dapi-test-token-123',
					}),
				}),
			);
		});
	});
});

describe('DatabricksVectorStore', () => {
	const mockFetchGlobal = mockFetch;

	const baseConfig = {
		workspaceUrl: 'https://dbc-test.cloud.databricks.com',
		token: 'dapi-test-token',
		indexName: 'catalog.schema.test_index',
		textColumn: 'content',
		metadataColumns: ['author'],
		scoreThreshold: 0.5,
	};

	const mockEmbeddings = {
		embedDocuments: jest.fn().mockResolvedValue([[0.1, 0.2, 0.3]]),
		embedQuery: jest.fn().mockResolvedValue([0.1, 0.2, 0.3]),
	};

	beforeEach(() => {
		jest.clearAllMocks();
	});

	describe('similaritySearchVectorWithScore', () => {
		it('should parse results using manifest column mapping', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					manifest: {
						column_count: 3,
						columns: [{ name: 'content' }, { name: 'author' }, { name: 'score' }],
					},
					result: {
						data_array: [
							['Hello world', 'Alice', 0.95],
							['Goodbye world', 'Bob', 0.82],
						],
						row_count: 2,
					},
				}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(
				mockEmbeddings as any,
				baseConfig,
			);
			const results = await store.similaritySearchVectorWithScore([0.1, 0.2, 0.3], 2);

			expect(results).toHaveLength(2);
			expect(results[0][0].pageContent).toBe('Hello world');
			expect(results[0][0].metadata.author).toBe('Alice');
			expect(results[0][1]).toBe(0.95);
			expect(results[1][0].pageContent).toBe('Goodbye world');
			expect(results[1][0].metadata.author).toBe('Bob');
			expect(results[1][1]).toBe(0.82);
		});

		it('should handle columns in any order via manifest mapping', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					manifest: {
						column_count: 3,
						columns: [{ name: 'score' }, { name: 'author' }, { name: 'content' }],
					},
					result: {
						data_array: [[0.9, 'Charlie', 'Test text']],
						row_count: 1,
					},
				}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(
				mockEmbeddings as any,
				baseConfig,
			);
			const results = await store.similaritySearchVectorWithScore([0.1], 1);

			expect(results[0][0].pageContent).toBe('Test text');
			expect(results[0][0].metadata.author).toBe('Charlie');
			expect(results[0][1]).toBe(0.9);
		});

		it('should return empty array when no results', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					manifest: { column_count: 2, columns: [{ name: 'content' }, { name: 'score' }] },
					result: { data_array: [], row_count: 0 },
				}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(
				mockEmbeddings as any,
				baseConfig,
			);
			const results = await store.similaritySearchVectorWithScore([0.1], 5);

			expect(results).toEqual([]);
		});

		it('should send correct request body with score threshold', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					manifest: { column_count: 2, columns: [{ name: 'content' }, { name: 'score' }] },
					result: { data_array: [], row_count: 0 },
				}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(
				mockEmbeddings as any,
				baseConfig,
			);
			await store.similaritySearchVectorWithScore([0.1, 0.2], 3);

			const callBody = JSON.parse(mockFetchGlobal.mock.calls[0][1].body);
			expect(callBody).toEqual({
				columns: ['content', 'author'],
				num_results: 3,
				query_vector: [0.1, 0.2],
				score_threshold: 0.5,
			});
		});

		it('should include filter in request when provided', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					manifest: { column_count: 1, columns: [{ name: 'content' }] },
					result: { data_array: [], row_count: 0 },
				}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(mockEmbeddings as any, {
				...baseConfig,
				metadataColumns: [],
				scoreThreshold: undefined,
			});
			await store.similaritySearchVectorWithScore([0.1], 1, { category: 'test' });

			const callBody = JSON.parse(mockFetchGlobal.mock.calls[0][1].body);
			expect(callBody.filters_json).toBe('{"category":"test"}');
		});

		it('should throw on API error', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: false,
				status: 404,
				text: async () => 'Index not found',
			});

			const store = await DatabricksVectorStore.fromExistingIndex(
				mockEmbeddings as any,
				baseConfig,
			);

			await expect(store.similaritySearchVectorWithScore([0.1], 1)).rejects.toThrow(
				'Databricks Vector Search API error (404): Index not found',
			);
		});

		it('should return score 0 when no score column in manifest', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					manifest: {
						column_count: 1,
						columns: [{ name: 'content' }],
					},
					result: {
						data_array: [['Some text']],
						row_count: 1,
					},
				}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(mockEmbeddings as any, {
				...baseConfig,
				metadataColumns: [],
			});
			const results = await store.similaritySearchVectorWithScore([0.1], 1);

			expect(results[0][1]).toBe(0);
		});
	});

	describe('addDocuments', () => {
		it('should embed documents and upsert to correct endpoint', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(
				mockEmbeddings as any,
				baseConfig,
			);
			await store.addDocuments([
				{ pageContent: 'test doc', metadata: { id: 'doc1', author: 'Alice' } },
			] as any);

			expect(mockEmbeddings.embedDocuments).toHaveBeenCalledWith(['test doc']);
			expect(mockFetchGlobal).toHaveBeenCalledWith(
				'https://dbc-test.cloud.databricks.com/api/2.0/vector-search/indexes/catalog.schema.test_index/upsert-data',
				expect.objectContaining({ method: 'POST' }),
			);
		});
	});

	describe('delete', () => {
		it('should call delete-data endpoint with primary keys', async () => {
			mockFetchGlobal.mockResolvedValueOnce({
				ok: true,
				json: async () => ({}),
			});

			const store = await DatabricksVectorStore.fromExistingIndex(
				mockEmbeddings as any,
				baseConfig,
			);
			await store.delete({ ids: ['doc1', 'doc2'] });

			expect(mockFetchGlobal).toHaveBeenCalledWith(
				'https://dbc-test.cloud.databricks.com/api/2.0/vector-search/indexes/catalog.schema.test_index/delete-data',
				expect.objectContaining({
					method: 'POST',
					body: JSON.stringify({ primary_keys: ['doc1', 'doc2'] }),
				}),
			);
		});
	});
});
