import { Embeddings } from '@langchain/core/embeddings';
import { logWrapper, getConnectionHintNoticeField } from '@n8n/ai-utilities';
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

import {
	validateDatabricksHost,
	validateResourceName,
	sanitizeErrorMessage,
} from '../../agents/DatabricksAiAgent/databricks-utils';

interface DatabricksCredential {
	host: string;
	token: string;
}

class DatabricksEmbeddings extends Embeddings {
	private readonly host: string;

	private readonly endpoint: string;

	private readonly apiKey: string;

	constructor(fields: { apiKey: string; host: string; endpoint: string }) {
		super({});
		this.apiKey = fields.apiKey;
		this.host = fields.host;
		this.endpoint = fields.endpoint;
	}

	/** @internal Override serialization to exclude credentials */
	get lc_secrets(): { [key: string]: string } {
		return { apiKey: 'DATABRICKS_TOKEN' };
	}

	async embedDocuments(texts: string[]): Promise<number[][]> {
		const response = await fetch(`${this.host}/serving-endpoints/${this.endpoint}/invocations`, {
			method: 'POST',
			headers: {
				Authorization: `Bearer ${this.apiKey}`,
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ input: texts }),
			signal: AbortSignal.timeout(60_000),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(
				`Databricks embeddings API error (${response.status}): ${sanitizeErrorMessage(errorText)}`,
			);
		}

		const result = (await response.json()) as {
			data?: Array<{ embedding: number[] }>;
			predictions?: number[][];
		};

		if (Array.isArray(result.data)) {
			return result.data.map((item) => item.embedding);
		}
		if (Array.isArray(result.predictions)) {
			return result.predictions;
		}
		throw new Error('Unexpected Databricks embeddings API response format.');
	}

	async embedQuery(text: string): Promise<number[]> {
		const embeddings = await this.embedDocuments([text]);
		if (!embeddings || !embeddings[0]) {
			throw new Error('No embedding returned from Databricks API. Check your endpoint and input.');
		}
		return embeddings[0];
	}
}

export class EmbeddingsDatabricks implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Embeddings Databricks',
		name: 'embeddingsDatabricks',
		icon: { light: 'file:databricks.svg', dark: 'file:databricks.dark.svg' },
		group: ['transform'],
		version: [1],
		description: 'Use Databricks serving endpoints for text embeddings',
		defaults: {
			name: 'Embeddings Databricks',
		},
		credentials: [
			{
				name: 'databricksApi',
				required: true,
			},
		],
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Embeddings'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.databricks.com/en/machine-learning/model-serving/index.html',
					},
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiEmbedding],
		outputNames: ['Embeddings'],
		requestDefaults: {
			baseURL: '={{$credentials.host}}',
			headers: {
				Authorization: '=Bearer {{$credentials.token}}',
			},
		},
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiVectorStore]),
			{
				displayName: 'Make sure the vector store and embedding model have the same dimensionality.',
				name: 'notice',
				type: 'notice',
				default: '',
			},
			{
				displayName: 'Serving Endpoint',
				name: 'model',
				type: 'options',
				description: 'The Databricks serving endpoint to use for embeddings',
				typeOptions: {
					loadOptions: {
						routing: {
							request: {
								method: 'GET',
								url: '/api/2.0/serving-endpoints',
							},
							output: {
								postReceive: [
									{
										type: 'rootProperty',
										properties: {
											property: 'endpoints',
										},
									},
									{
										type: 'setKeyValue',
										properties: {
											name: '={{$responseItem.name}}',
											value: '={{$responseItem.name}}',
										},
									},
									{
										type: 'sort',
										properties: {
											key: 'name',
										},
									},
								],
							},
						},
					},
				},
				routing: {
					send: {
						type: 'body',
						property: 'model',
					},
				},
				default: '',
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials<DatabricksCredential>('databricksApi');
		const endpoint = this.getNodeParameter('model', itemIndex) as string;

		const host = validateDatabricksHost(credentials.host);
		const safeEndpoint = validateResourceName(endpoint, 'Serving endpoint');

		const embeddings = new DatabricksEmbeddings({
			apiKey: credentials.token,
			host,
			endpoint: safeEndpoint,
		});

		return {
			response: logWrapper(embeddings, this),
		};
	}
}
