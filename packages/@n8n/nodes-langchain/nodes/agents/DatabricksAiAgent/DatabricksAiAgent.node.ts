import { NodeConnectionTypes } from 'n8n-workflow';
import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	EngineResponse,
	EngineRequest,
} from 'n8n-workflow';

import type { RequestResponseMetadata } from '@utils/agent-execution';
import {
	promptTypeOptions,
	textFromGuardrailsNode,
	textFromPreviousNode,
	textInput,
} from '@utils/descriptions';

import { toolsAgentProperties } from '../Agent/agents/ToolsAgent/V3/description';
import { toolsAgentExecute } from '../Agent/agents/ToolsAgent/V3/execute';
import { getInputs } from '../Agent/utils';

export class DatabricksAiAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Databricks AI Agent',
		name: 'databricksAiAgent',
		icon: { light: 'file:databricks.svg', dark: 'file:databricks.dark.svg' },
		group: ['transform'],
		version: [1],
		description:
			'Generates an action plan and executes it using a Databricks chat model. Can use external tools.',
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
						url: 'https://docs.databricks.com/en/machine-learning/index.html',
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
		builderHint: {
			inputs: {
				ai_languageModel: { required: true },
				ai_memory: { required: false },
				ai_tool: { required: false },
				ai_outputParser: {
					required: false,
					displayOptions: { show: { hasOutputParser: [true] } },
				},
			},
		},
		properties: [
			promptTypeOptions,
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
		hints: [
			{
				message:
					'You are using streaming responses. Make sure to set the response mode to "Streaming Response" on the connected trigger node.',
				type: 'warning',
				location: 'outputPane',
				whenToDisplay: 'afterExecution',
				displayCondition: '={{ $parameter["enableStreaming"] === true }}',
			},
		],
	};

	async execute(
		this: IExecuteFunctions,
		response?: EngineResponse<RequestResponseMetadata>,
	): Promise<INodeExecutionData[][] | EngineRequest<RequestResponseMetadata>> {
		return await toolsAgentExecute.call(this, response);
	}
}
