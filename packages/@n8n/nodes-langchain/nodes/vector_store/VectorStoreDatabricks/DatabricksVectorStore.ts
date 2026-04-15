import { VectorStore } from '@langchain/core/vectorstores';
import type { Embeddings } from '@langchain/core/embeddings';
import { Document } from '@langchain/core/documents';

import { sanitizeErrorMessage } from '../../agents/DatabricksAiAgent/databricks-utils';

export interface DatabricksVectorStoreConfig {
	workspaceUrl: string;
	token: string;
	indexName: string;
	textColumn: string;
	metadataColumns: string[];
	scoreThreshold?: number;
}

interface DatabricksVectorSearchResponse {
	manifest: {
		column_count: number;
		columns: Array<{ name: string }>;
	};
	next_page_token?: string;
	result: {
		data_array: Array<Array<string | number | number[]>>;
		row_count: number;
	};
}

export class DatabricksVectorStore extends VectorStore {
	private readonly config: DatabricksVectorStoreConfig;

	_vectorstoreType(): string {
		return 'databricks';
	}

	constructor(embeddings: Embeddings, config: DatabricksVectorStoreConfig) {
		super(embeddings, {});
		this.config = config;
	}

	/** @internal Override serialization to exclude credentials */
	get lc_secrets(): { [key: string]: string } {
		return { 'config.token': 'DATABRICKS_TOKEN' };
	}

	private async makeRequest(
		body: Record<string, unknown>,
	): Promise<DatabricksVectorSearchResponse> {
		const url = `${this.config.workspaceUrl}/api/2.0/vector-search/indexes/${this.config.indexName}/query`;

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				Authorization: `Bearer ${this.config.token}`,
				'Content-Type': 'application/json',
				Accept: 'application/json',
			},
			body: JSON.stringify(body),
			signal: AbortSignal.timeout(60_000),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(
				`Databricks Vector Search API error (${response.status}): ${sanitizeErrorMessage(errorText)}`,
			);
		}

		return (await response.json()) as DatabricksVectorSearchResponse;
	}

	static async fromDocuments(
		docs: Document[],
		embeddings: Embeddings,
		config: DatabricksVectorStoreConfig,
	): Promise<DatabricksVectorStore> {
		const instance = new this(embeddings, config);
		await instance.addDocuments(docs);
		return instance;
	}

	static async fromExistingIndex(
		embeddings: Embeddings,
		config: DatabricksVectorStoreConfig,
	): Promise<DatabricksVectorStore> {
		return new this(embeddings, config);
	}

	async addDocuments(documents: Document[]): Promise<void> {
		const texts = documents.map((doc) => doc.pageContent);
		const vectors = await this.embeddings.embedDocuments(texts);
		await this.addVectors(vectors, documents);
	}

	async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
		const url = `${this.config.workspaceUrl}/api/2.0/vector-search/indexes/${this.config.indexName}/upsert-data`;

		const rows = vectors.map((vector, i) => {
			const row: Record<string, unknown> = {
				id: (documents[i].metadata?.id as string) || `doc_${i}`,
				embedding: vector,
				[this.config.textColumn]: documents[i].pageContent,
			};
			for (const col of this.config.metadataColumns) {
				if (documents[i].metadata?.[col] !== undefined) {
					row[col] = documents[i].metadata[col];
				}
			}
			return row;
		});

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				Authorization: `Bearer ${this.config.token}`,
				'Content-Type': 'application/json',
				Accept: 'application/json',
			},
			body: JSON.stringify({ inputs_json: JSON.stringify(rows) }),
			signal: AbortSignal.timeout(60_000),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(
				`Databricks upsert error (${response.status}): ${sanitizeErrorMessage(errorText)}`,
			);
		}
	}

	async delete(params: { ids: string[] }): Promise<void> {
		const url = `${this.config.workspaceUrl}/api/2.0/vector-search/indexes/${this.config.indexName}/delete-data`;

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				Authorization: `Bearer ${this.config.token}`,
				'Content-Type': 'application/json',
				Accept: 'application/json',
			},
			body: JSON.stringify({ primary_keys: params.ids }),
			signal: AbortSignal.timeout(60_000),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(
				`Databricks delete error (${response.status}): ${sanitizeErrorMessage(errorText)}`,
			);
		}
	}

	async similaritySearchVectorWithScore(
		query: number[],
		k: number,
		filter?: Record<string, unknown>,
	): Promise<[Document, number][]> {
		// Build columns: text column + metadata columns
		const columns = [this.config.textColumn, ...this.config.metadataColumns];

		const body: Record<string, unknown> = {
			columns,
			num_results: k,
			query_vector: query,
		};

		if (filter && Object.keys(filter).length > 0) {
			body.filters_json = JSON.stringify(filter);
		}

		if (this.config.scoreThreshold !== undefined) {
			body.score_threshold = this.config.scoreThreshold;
		}

		const response = await this.makeRequest(body);

		if (!response?.result?.data_array || !Array.isArray(response.result.data_array)) {
			return [];
		}

		// Build column name → index mapping from the manifest
		const columnMap = new Map<string, number>();
		for (let i = 0; i < response.manifest.columns.length; i++) {
			columnMap.set(response.manifest.columns[i].name, i);
		}

		const textColIdx = columnMap.get(this.config.textColumn);
		const scoreColIdx = columnMap.get('score');

		return response.result.data_array.map((row) => {
			const pageContent = textColIdx !== undefined ? String(row[textColIdx]) : '';
			const score = scoreColIdx !== undefined ? Number(row[scoreColIdx]) : 0;

			// Extract metadata from the row using column mapping
			const metadata: Record<string, unknown> = {};
			for (const col of this.config.metadataColumns) {
				const idx = columnMap.get(col);
				if (idx !== undefined) {
					metadata[col] = row[idx];
				}
			}

			const doc = new Document({ pageContent, metadata });
			return [doc, score];
		});
	}
}
