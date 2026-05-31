import { createVectorStoreNode, metadataFilterField } from '@n8n/ai-utilities';
import { NodeOperationError, type INodeProperties } from 'n8n-workflow';

import { DatabricksVectorStore } from './DatabricksVectorStore';
import {
	validateDatabricksHost,
	validateResourceName,
} from '../../vendors/Databricks/databricks-utils';

interface DatabricksCredential {
	host: string;
	token: string;
}

const indexNameField: INodeProperties = {
	displayName: 'Index Name',
	name: 'indexName',
	type: 'string',
	default: '',
	required: true,
	placeholder: 'catalog.schema.my_index',
	description: 'The Databricks Vector Search index to query (three-level namespace)',
};

const sharedFields: INodeProperties[] = [indexNameField];

const retrieveFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [
			{
				displayName: 'Text Column',
				name: 'textColumn',
				type: 'string',
				default: 'text',
				description: 'The index column that holds the document text',
			},
			{
				displayName: 'Metadata Columns',
				name: 'metadataColumns',
				type: 'string',
				default: '',
				placeholder: 'source,title,url',
				description: 'Comma-separated list of additional columns to return as metadata',
			},
			{
				displayName: 'Score Threshold',
				name: 'scoreThreshold',
				type: 'number',
				default: 0,
				typeOptions: { minValue: 0, maxValue: 1, numberPrecision: 2 },
				description: 'Minimum relevance score a result must meet to be returned',
			},
			metadataFilterField,
		],
	},
];

function buildConfig(
	options: { textColumn?: string; metadataColumns?: string; scoreThreshold?: number },
	workspaceUrl: string,
	token: string,
	indexName: string,
	filter?: Record<string, never>,
) {
	return {
		workspaceUrl,
		token,
		indexName,
		textColumn: options.textColumn ?? 'text',
		metadataColumns: (options.metadataColumns ?? '')
			.split(',')
			.map((column) => column.trim())
			.filter((column) => column.length > 0),
		scoreThreshold: options.scoreThreshold ? options.scoreThreshold : undefined,
		filter,
	};
}

export class VectorStoreDatabricks extends createVectorStoreNode<DatabricksVectorStore>({
	meta: {
		displayName: 'Databricks Vector Store',
		name: 'vectorStoreDatabricks',
		description: 'Query a Databricks Vector Search index',
		icon: { light: 'file:databricks.svg', dark: 'file:databricks.dark.svg' },
		docsUrl: 'https://docs.databricks.com/en/generative-ai/vector-search.html',
		credentials: [
			{
				name: 'databricksApi',
				required: true,
			},
		],
		operationModes: ['load', 'retrieve', 'retrieve-as-tool'],
	},
	sharedFields,
	loadFields: retrieveFields,
	retrieveFields,
	async getVectorStoreClient(context, filter, embeddings, itemIndex) {
		const indexName = context.getNodeParameter('indexName', itemIndex, '') as string;
		validateResourceName(indexName, 'Vector Search index');

		const options = context.getNodeParameter('options', itemIndex, {}) as {
			textColumn?: string;
			metadataColumns?: string;
			scoreThreshold?: number;
		};

		const credentials = await context.getCredentials<DatabricksCredential>('databricksApi');
		const workspaceUrl = await validateDatabricksHost(credentials.host);

		return await DatabricksVectorStore.fromExistingIndex(
			embeddings,
			buildConfig(options, workspaceUrl, credentials.token, indexName, filter),
		);
	},
	async populateVectorStore(context) {
		// 'insert' is intentionally not in operationModes — Databricks indexes are
		// populated and synced within Databricks, not from n8n.
		throw new NodeOperationError(
			context.getNode(),
			'Databricks Vector Search indexes are populated within Databricks. Inserting documents from n8n is not supported.',
		);
	},
}) {}
