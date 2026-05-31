import type { Embeddings } from '@langchain/core/embeddings';
import { proxyFetch } from '@n8n/ai-utilities';
import type { Mock } from 'vitest';

import { DatabricksVectorStore, type DatabricksVectorStoreConfig } from '../DatabricksVectorStore';

vi.mock('@n8n/ai-utilities', async () => {
	const actual = await vi.importActual('@n8n/ai-utilities');
	return {
		...actual,
		proxyFetch: vi.fn(),
	};
});

const mockedProxyFetch = vi.mocked(proxyFetch) as unknown as Mock;

const fakeEmbeddings = {
	embedQuery: vi.fn(),
	embedDocuments: vi.fn(),
} as unknown as Embeddings;

const baseConfig: DatabricksVectorStoreConfig = {
	workspaceUrl: 'https://dbc-test.cloud.databricks.com',
	token: 'dapi-test',
	indexName: 'catalog.schema.my_index',
	textColumn: 'text',
	metadataColumns: ['source'],
};

const queryResponse = (rows: unknown[][]) => ({
	ok: true,
	json: async () => ({
		manifest: { columns: [{ name: 'text' }, { name: 'source' }, { name: 'score' }] },
		result: { data_array: rows },
	}),
});

describe('DatabricksVectorStore', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	it('should reject inserting documents', async () => {
		const store = new DatabricksVectorStore(fakeEmbeddings, baseConfig);
		await expect(store.addDocuments()).rejects.toThrow('not supported');
		await expect(store.addVectors()).rejects.toThrow('not supported');
	});

	it('should query the vector-search endpoint and map results to documents with scores', async () => {
		mockedProxyFetch.mockResolvedValue(
			queryResponse([
				['hello world', 'doc-a', 0.92],
				['second result', 'doc-b', 0.81],
			]),
		);

		const store = new DatabricksVectorStore(fakeEmbeddings, baseConfig);
		const results = await store.similaritySearchVectorWithScore([0.1, 0.2, 0.3], 5);

		// Correct endpoint + auth
		const [url, init] = mockedProxyFetch.mock.calls[0];
		expect(url).toBe(
			'https://dbc-test.cloud.databricks.com/api/2.0/vector-search/indexes/catalog.schema.my_index/query',
		);
		expect(init.headers.Authorization).toBe('Bearer dapi-test');
		const body = JSON.parse(init.body);
		expect(body.columns).toEqual(['text', 'source']);
		expect(body.query_vector).toEqual([0.1, 0.2, 0.3]);

		// Mapped documents + scores
		expect(results).toHaveLength(2);
		expect(results[0][0].pageContent).toBe('hello world');
		expect(results[0][0].metadata).toEqual({ source: 'doc-a' });
		expect(results[0][1]).toBe(0.92);
	});

	it('should limit returned results to k', async () => {
		mockedProxyFetch.mockResolvedValue(
			queryResponse([
				['a', 's1', 0.9],
				['b', 's2', 0.8],
				['c', 's3', 0.7],
			]),
		);

		const store = new DatabricksVectorStore(fakeEmbeddings, baseConfig);
		const results = await store.similaritySearchVectorWithScore([0.1], 2);

		expect(results).toHaveLength(2);
	});

	it('should forward a metadata filter as filters_json and the score threshold', async () => {
		mockedProxyFetch.mockResolvedValue(queryResponse([]));

		const store = new DatabricksVectorStore(fakeEmbeddings, {
			...baseConfig,
			scoreThreshold: 0.5,
		});
		await store.similaritySearchVectorWithScore([0.1], 3, { source: 'docs' });

		const body = JSON.parse(mockedProxyFetch.mock.calls[0][1].body);
		expect(body.filters_json).toBe(JSON.stringify({ source: 'docs' }));
		expect(body.score_threshold).toBe(0.5);
	});

	it('should throw a descriptive error on a non-ok response', async () => {
		mockedProxyFetch.mockResolvedValue({
			ok: false,
			status: 403,
			text: async () => 'permission denied',
		});

		const store = new DatabricksVectorStore(fakeEmbeddings, baseConfig);
		await expect(store.similaritySearchVectorWithScore([0.1], 3)).rejects.toThrow(
			'Databricks Vector Search request failed (403)',
		);
	});
});
