import type { INodeProperties } from 'n8n-workflow';

import { createVectorStoreNode } from '@n8n/ai-utilities';

import { DatabricksVectorStore } from './DatabricksVectorStore';
import {
	validateDatabricksHost,
	validateResourceName,
} from '../../agents/DatabricksAiAgent/databricks-utils';

const sharedFields: INodeProperties[] = [
	{
		displayName: 'Index Name',
		name: 'indexName',
		type: 'string',
		default: '',
		required: true,
		placeholder: 'e.g. catalog.schema.my_vector_index',
		description: 'Full name of the Databricks Vector Search index',
	},
	{
		displayName: 'Text Column',
		name: 'textColumn',
		type: 'string',
		default: 'text',
		required: true,
		description: 'Name of the column containing the document text',
	},
	{
		displayName: 'Metadata Columns',
		name: 'metadataColumns',
		type: 'string',
		default: '',
		description: 'Comma-separated list of columns to include as metadata',
	},
];

const retrieveFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [
			{
				displayName: 'Score Threshold',
				name: 'scoreThreshold',
				type: 'number',
				default: 0,
				typeOptions: { maxValue: 1, minValue: 0, numberPrecision: 2 },
				description: 'Minimum similarity score threshold (0 = no threshold)',
			},
		],
	},
];

export class VectorStoreDatabricks extends createVectorStoreNode<DatabricksVectorStore>({
	meta: {
		displayName: 'Databricks Vector Store',
		name: 'vectorStoreDatabricks',
		description: 'Work with your data in Databricks Vector Search',
		icon: { light: 'file:databricks.svg', dark: 'file:databricks.dark.svg' },
		docsUrl: 'https://docs.databricks.com/en/generative-ai/vector-search.html',
		credentials: [
			{
				name: 'databricksApi',
				required: true,
			},
		],
		operationModes: ['load', 'insert', 'retrieve', 'retrieve-as-tool'],
	},
	sharedFields,
	retrieveFields,
	loadFields: retrieveFields,
	insertFields: retrieveFields,
	async getVectorStoreClient(context, _filter, embeddings, itemIndex) {
		const credentials = await context.getCredentials('databricksApi');
		const indexName = context.getNodeParameter('indexName', itemIndex) as string;
		const textColumn = context.getNodeParameter('textColumn', itemIndex) as string;
		const metadataColumnsRaw = context.getNodeParameter('metadataColumns', itemIndex, '') as string;
		const options = context.getNodeParameter('options', itemIndex, {}) as {
			scoreThreshold?: number;
		};

		const host = validateDatabricksHost(credentials.host as string);
		const safeIndexName = validateResourceName(indexName, 'Index');
		const safeTextColumn = validateResourceName(textColumn, 'Text column');

		const metadataColumns = metadataColumnsRaw
			? metadataColumnsRaw
					.split(',')
					.map((s: string) => s.trim())
					.filter(Boolean)
					.map((col) => validateResourceName(col, 'Metadata column'))
			: [];

		return DatabricksVectorStore.fromExistingIndex(embeddings, {
			workspaceUrl: host,
			token: credentials.token as string,
			indexName: safeIndexName,
			textColumn: safeTextColumn,
			metadataColumns,
			scoreThreshold: options.scoreThreshold || undefined,
		});
	},
	async populateVectorStore(context, embeddings, documents, itemIndex) {
		const credentials = await context.getCredentials('databricksApi');
		const indexName = context.getNodeParameter('indexName', itemIndex) as string;
		const textColumn = context.getNodeParameter('textColumn', itemIndex) as string;
		const metadataColumnsRaw = context.getNodeParameter('metadataColumns', itemIndex, '') as string;

		const host = validateDatabricksHost(credentials.host as string);
		const safeIndexName = validateResourceName(indexName, 'Index');
		const safeTextColumn = validateResourceName(textColumn, 'Text column');

		const metadataColumns = metadataColumnsRaw
			? metadataColumnsRaw
					.split(',')
					.map((s: string) => s.trim())
					.filter(Boolean)
					.map((col) => validateResourceName(col, 'Metadata column'))
			: [];

		await DatabricksVectorStore.fromDocuments(documents, embeddings, {
			workspaceUrl: host,
			token: credentials.token as string,
			indexName: safeIndexName,
			textColumn: safeTextColumn,
			metadataColumns,
		});
	},
}) {}
