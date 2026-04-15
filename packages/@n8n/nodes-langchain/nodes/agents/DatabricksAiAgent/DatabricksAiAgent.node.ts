import { NodeConnectionTypes } from 'n8n-workflow';
import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	EngineResponse,
	EngineRequest,
} from 'n8n-workflow';

import type { RequestResponseMetadata } from '../../../utils/agent-execution';
import {
	promptTypeOptions,
	textFromPreviousNode,
	textInput,
	textFromGuardrailsNode,
} from '../../../utils/descriptions';

import { toolsAgentProperties } from '../Agent/agents/ToolsAgent/V3/description';
import { toolsAgentExecute } from '../Agent/agents/ToolsAgent/V3/execute';
import { getInputs } from '../Agent/utils';
import { setupMlflowTracing } from './mlflow-utils';

export class DatabricksAiAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Databricks AI Agent',
		name: 'databricksAiAgent',
		icon: { light: 'file:databricks.svg', dark: 'file:databricks.dark.svg' },
		group: ['transform'],
		version: [1],
		description:
			'Databricks AI Agent with MLflow tracking. Generates an action plan and executes it. Can use external tools.',
		defaults: {
			name: 'Databricks AI Agent',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Agents', 'Root Nodes'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.databricks.com/en/mlflow/index.html',
					},
				],
			},
		},
		inputs: `={{
			((hasOutputParser, needsFallback) => {
				${getInputs.toString()};
				return getInputs(true, hasOutputParser, needsFallback);
			})($parameter.hasOutputParser === undefined || $parameter.hasOutputParser === true, $parameter.needsFallback !== undefined && $parameter.needsFallback === true)
		}}`,
		outputs: [NodeConnectionTypes.Main],
		credentials: [
			{
				name: 'databricksApi',
				required: false,
			},
		],
		properties: [
			{
				displayName: 'Enable MLflow Tracking',
				name: 'enableMlflow',
				type: 'boolean',
				default: false,
				description:
					'Whether to log agent execution traces (input requests, AI reasoning, tool calls, output responses) to managed MLflow in your Databricks workspace',
			},
			{
				displayName:
					'When MLflow is enabled, traces will be logged to experiment /Shared/n8n-workflows-{workflowId} in your Databricks workspace. Make sure Databricks credentials are configured.',
				name: 'mlflowNotice',
				type: 'notice',
				default: '',
				displayOptions: {
					show: {
						enableMlflow: [true],
					},
				},
			},
			{
				...promptTypeOptions,
			},
			{
				...textFromGuardrailsNode,
				displayOptions: {
					show: {
						promptType: ['guardrails'],
					},
				},
			},
			{
				...textFromPreviousNode,
				displayOptions: {
					show: {
						promptType: ['auto'],
					},
				},
			},
			{
				...textInput,
				displayOptions: {
					show: {
						promptType: ['define'],
					},
				},
			},
			{
				displayName: 'Require Specific Output Format',
				name: 'hasOutputParser',
				type: 'boolean',
				default: false,
				noDataExpression: true,
			},
			{
				displayName: `Connect an <a data-action='openSelectiveNodeCreator' data-action-parameter-connectiontype='${NodeConnectionTypes.AiOutputParser}'>output parser</a> on the canvas to specify the output format you require`,
				name: 'notice',
				type: 'notice',
				default: '',
				displayOptions: {
					show: {
						hasOutputParser: [true],
					},
				},
			},
			{
				displayName: 'Enable Fallback Model',
				name: 'needsFallback',
				type: 'boolean',
				default: false,
				noDataExpression: true,
			},
			{
				displayName:
					'Connect an additional language model on the canvas to use it as a fallback if the main model fails',
				name: 'fallbackNotice',
				type: 'notice',
				default: '',
				displayOptions: {
					show: {
						needsFallback: [true],
					},
				},
			},
			toolsAgentProperties,
		],
	};

	async execute(
		this: IExecuteFunctions,
		response?: EngineResponse<RequestResponseMetadata>,
	): Promise<INodeExecutionData[][] | EngineRequest<RequestResponseMetadata>> {
		// Set up MLflow tracing if enabled (only on first call, not on tool call continuations)
		if (!response) {
			this.logger.debug('Setting up Databricks AI Agent execution');
			try {
				const mlflowHandler = await setupMlflowTracing(this);
				if (mlflowHandler) {
					this.logger.info('MLflow tracing enabled for this execution');
					// The MLflow handler is initialized globally via mlflow.init()
					// and will capture spans from all LangChain operations in this execution
				}
			} catch (error) {
				this.logger.warn(
					`MLflow setup failed, continuing without tracing: ${error instanceof Error ? error.message : String(error)}`,
				);
			}
		}

		// Delegate to the standard Tools Agent V3 executor
		return await toolsAgentExecute.call(this, response);
	}
}
