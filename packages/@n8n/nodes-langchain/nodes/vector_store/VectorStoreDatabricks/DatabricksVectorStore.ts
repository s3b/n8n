import { Document } from '@langchain/core/documents';
import type { Embeddings } from '@langchain/core/embeddings';
import { VectorStore } from '@langchain/core/vectorstores';
import { proxyFetch } from '@n8n/ai-utilities';
import { OperationalError } from 'n8n-workflow';

export interface DatabricksVectorStoreConfig {
	/** Validated, public HTTPS Databricks workspace URL (no trailing slash). */
	workspaceUrl: string;
	token: string;
	indexName: string;
	textColumn: string;
	metadataColumns: string[];
	scoreThreshold?: number;
	/** Metadata filter forwarded to the index as `filters_json`. */
	filter?: Record<string, unknown>;
}

interface DatabricksQueryResponse {
	manifest?: {
		columns?: Array<{ name: string }>;
	};
	result?: {
		data_array?: unknown[][];
	};
}

/**
 * Minimal LangChain `VectorStore` backed by the Databricks Vector Search query API.
 *
 * Databricks indexes are created and synced inside Databricks, so this store is
 * read-only: it implements similarity search against
 * `/api/2.0/vector-search/indexes/{index}/query` and rejects write operations.
 */
export class DatabricksVectorStore extends VectorStore {
	private readonly config: DatabricksVectorStoreConfig;

	_vectorstoreType(): string {
		return 'databricks';
	}

	constructor(embeddings: Embeddings, config: DatabricksVectorStoreConfig) {
		super(embeddings, {});
		this.config = config;
	}

	static async fromExistingIndex(
		embeddings: Embeddings,
		config: DatabricksVectorStoreConfig,
	): Promise<DatabricksVectorStore> {
		return new DatabricksVectorStore(embeddings, config);
	}

	async addDocuments(): Promise<void> {
		throw new OperationalError(
			'Databricks Vector Search indexes are populated and synced within Databricks. Inserting documents from n8n is not supported.',
		);
	}

	async addVectors(): Promise<void> {
		throw new OperationalError(
			'Databricks Vector Search indexes are populated and synced within Databricks. Inserting vectors from n8n is not supported.',
		);
	}

	private async query(queryVector: number[]): Promise<DatabricksQueryResponse> {
		const { workspaceUrl, indexName, token, textColumn, metadataColumns, scoreThreshold, filter } =
			this.config;

		const columns = [textColumn, ...metadataColumns.filter((column) => column !== textColumn)];

		const body: Record<string, unknown> = {
			columns,
			query_vector: queryVector,
		};
		if (scoreThreshold !== undefined) body.score_threshold = scoreThreshold;
		if (filter && Object.keys(filter).length > 0) body.filters_json = JSON.stringify(filter);

		const response = await proxyFetch(
			`${workspaceUrl}/api/2.0/vector-search/indexes/${indexName}/query`,
			{
				method: 'POST',
				headers: {
					Authorization: `Bearer ${token}`,
					'Content-Type': 'application/json',
					Accept: 'application/json',
				},
				body: JSON.stringify(body),
			},
		);

		if (!response.ok) {
			const detail = await response.text();
			throw new OperationalError(
				`Databricks Vector Search request failed (${response.status}): ${detail.slice(0, 500)}`,
			);
		}

		return (await response.json()) as DatabricksQueryResponse;
	}

	async similaritySearchVectorWithScore(
		query: number[],
		k: number,
		filter?: Record<string, unknown>,
	): Promise<Array<[Document, number]>> {
		if (filter && Object.keys(filter).length > 0) {
			this.config.filter = filter;
		}

		const response = await this.query(query);

		const columnNames = (response.manifest?.columns ?? []).map((column) => column.name);
		const rows = response.result?.data_array ?? [];

		return rows.slice(0, k).map((row): [Document, number] => {
			// Zip the row values against the manifest column names.
			const record: Record<string, unknown> = {};
			columnNames.forEach((name, index) => {
				record[name] = row[index];
			});

			const pageContent = String(record[this.config.textColumn] ?? '');

			const metadata: Record<string, unknown> = {};
			for (const column of this.config.metadataColumns) {
				if (column in record) metadata[column] = record[column];
			}

			// Databricks returns the relevance score in a `score` / `__db_score` column.
			const rawScore = record.score ?? record.__db_score;
			const score = typeof rawScore === 'number' ? rawScore : 0;

			return [new Document({ pageContent, metadata }), score];
		});
	}
}
