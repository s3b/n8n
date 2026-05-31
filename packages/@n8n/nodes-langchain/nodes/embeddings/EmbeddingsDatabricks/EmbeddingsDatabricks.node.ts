import { OpenAIEmbeddings, type ClientOptions } from '@langchain/openai';
import { getProxyAgent, logWrapper, getConnectionHintNoticeField } from '@n8n/ai-utilities';
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

import {
	loadDatabricksServingEndpoints,
	validateDatabricksHost,
	validateResourceName,
} from '../../vendors/Databricks/databricks-utils';

interface DatabricksCredential {
	host: string;
	token: string;
}

export class EmbeddingsDatabricks implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Embeddings Databricks',
		name: 'embeddingsDatabricks',
		icon: { light: 'file:databricks.svg', dark: 'file:databricks.dark.svg' },
		group: ['transform'],
		version: [1],
		description: 'Use Databricks Model Serving embeddings endpoints',
		defaults: {
			name: 'Embeddings Databricks',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Embeddings'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.databricks.com/en/machine-learning/foundation-models/index.html',
					},
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiEmbedding],
		outputNames: ['Embeddings'],
		credentials: [
			{
				name: 'databricksApi',
				required: true,
			},
		],
		requestDefaults: {
			baseURL: '={{$credentials.host}}',
			headers: {
				Authorization: '=Bearer {{$credentials.token}}',
			},
		},
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiVectorStore]),
			{
				displayName: 'Model Name or ID',
				name: 'model',
				type: 'options',
				description:
					'The Databricks serving endpoint to use for embeddings. Choose from the list, or specify an ID using an <a href="https://docs.n8n.io/code/expressions/">expression</a>.',
				typeOptions: {
					loadOptionsMethod: 'getModels',
				},
				default: '',
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description: 'Additional options to add',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Batch Size',
						name: 'batchSize',
						default: 512,
						typeOptions: { maxValue: 2048 },
						description: 'Maximum number of documents to send in each request',
						type: 'number',
					},
					{
						displayName: 'Strip New Lines',
						name: 'stripNewLines',
						default: true,
						description: 'Whether to strip new lines from the input text',
						type: 'boolean',
					},
					{
						displayName: 'Timeout',
						name: 'timeout',
						default: 60000,
						description: 'Maximum amount of time a request is allowed to take in milliseconds',
						type: 'number',
					},
				],
			},
		],
	};

	methods = {
		loadOptions: {
			getModels: loadDatabricksServingEndpoints,
		},
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials<DatabricksCredential>('databricksApi');

		const modelName = this.getNodeParameter('model', itemIndex) as string;

		const options = this.getNodeParameter('options', itemIndex, {}) as {
			batchSize?: number;
			stripNewLines?: boolean;
			timeout?: number;
		};

		const timeout = options.timeout ?? 60000;
		const host = await validateDatabricksHost(credentials.host);
		validateResourceName(modelName, 'Embeddings endpoint');
		const baseURL = `${host}/serving-endpoints`;

		const configuration: ClientOptions = {
			baseURL,
			fetchOptions: {
				dispatcher: getProxyAgent(baseURL, {
					headersTimeout: timeout,
					bodyTimeout: timeout,
				}),
			},
		};

		const embeddings = new OpenAIEmbeddings({
			apiKey: credentials.token,
			model: modelName,
			batchSize: options.batchSize,
			stripNewLines: options.stripNewLines,
			timeout,
			configuration,
			// Databricks returns embeddings as a plain float array. The OpenAI SDK
			// otherwise defaults to requesting base64 and mis-decodes that array into
			// a garbled, wrong-length vector, so request the float format explicitly.
			encodingFormat: 'float',
		});

		return {
			response: logWrapper(embeddings, this),
		};
	}
}
