import type { IExecuteFunctions } from 'n8n-workflow';
import { mock } from 'jest-mock-extended';

jest.mock('./mlflow-utils', () => ({
	setupMlflowTracing: jest.fn(),
}));

jest.mock('../Agent/agents/ToolsAgent/V3/execute', () => ({
	toolsAgentExecute: jest.fn().mockResolvedValue([[{ json: { output: 'test' } }]]),
}));

jest.mock('../Agent/agents/ToolsAgent/V3/description', () => ({
	toolsAgentProperties: {
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		default: {},
	},
}));

jest.mock('../Agent/utils', () => ({
	getInputs: jest.fn().mockReturnValue([]),
}));

jest.mock('../../../utils/descriptions', () => ({
	promptTypeOptions: {
		displayName: 'Prompt',
		name: 'promptType',
		type: 'options',
		default: 'auto',
	},
	promptTypeOptionsDeprecated: {
		displayName: 'Prompt',
		name: 'promptType',
		type: 'options',
		default: 'auto',
	},
	textFromPreviousNode: { displayName: 'Text', name: 'text', type: 'string', default: '' },
	textInput: { displayName: 'Text', name: 'text', type: 'string', default: '' },
	textFromGuardrailsNode: { displayName: 'Text', name: 'text', type: 'string', default: '' },
}));

jest.mock('../../../utils/agent-execution', () => ({
	RequestResponseMetadata: {},
}));

import { DatabricksAiAgent } from './DatabricksAiAgent.node';
import { setupMlflowTracing } from './mlflow-utils';
import { toolsAgentExecute } from '../Agent/agents/ToolsAgent/V3/execute';
import { MlflowCallbackHandler } from './MlflowCallbackHandler';

const mockedSetupMlflow = setupMlflowTracing as jest.MockedFunction<typeof setupMlflowTracing>;
const mockedToolsAgentExecute = toolsAgentExecute as jest.MockedFunction<typeof toolsAgentExecute>;

describe('DatabricksAiAgent', () => {
	let agent: DatabricksAiAgent;

	const createMockContext = () => {
		const ctx = mock<IExecuteFunctions>();
		ctx.getNodeParameter = jest.fn().mockReturnValue(false);
		ctx.logger = {
			debug: jest.fn(),
			info: jest.fn(),
			warn: jest.fn(),
			error: jest.fn(),
			verbose: jest.fn(),
		} as any;
		return ctx;
	};

	beforeEach(() => {
		agent = new DatabricksAiAgent();
		jest.clearAllMocks();
	});

	describe('description', () => {
		it('should have correct node metadata', () => {
			expect(agent.description.name).toBe('databricksAiAgent');
			expect(agent.description.displayName).toBe('Databricks AI Agent');
			expect(agent.description.credentials).toEqual([{ name: 'databricksApi', required: false }]);
		});

		it('should have enableMlflow property', () => {
			const mlflowProp = agent.description.properties.find((p) => p.name === 'enableMlflow');
			expect(mlflowProp).toBeDefined();
			expect(mlflowProp?.type).toBe('boolean');
			expect(mlflowProp?.default).toBe(false);
		});

		it('should have notice shown when MLflow is enabled', () => {
			const noticeProp = agent.description.properties.find((p) => p.name === 'mlflowNotice');
			expect(noticeProp).toBeDefined();
			expect(noticeProp?.displayOptions?.show?.enableMlflow).toEqual([true]);
		});
	});

	describe('execute', () => {
		it('should set up MLflow tracing and delegate to toolsAgentExecute', async () => {
			const ctx = createMockContext();
			mockedSetupMlflow.mockResolvedValue(new MlflowCallbackHandler());

			await agent.execute.call(ctx);

			expect(mockedSetupMlflow).toHaveBeenCalledWith(ctx);
			expect(mockedToolsAgentExecute).toHaveBeenCalled();
		});

		it('should skip MLflow setup on continuation (response present)', async () => {
			const ctx = createMockContext();
			const response = { actions: [], metadata: {} } as any;

			await agent.execute.call(ctx, response);

			expect(mockedSetupMlflow).not.toHaveBeenCalled();
			expect(mockedToolsAgentExecute).toHaveBeenCalledWith(response);
		});

		it('should continue execution even if MLflow setup fails', async () => {
			const ctx = createMockContext();
			mockedSetupMlflow.mockRejectedValue(new Error('MLflow init failed'));

			await agent.execute.call(ctx);

			expect(ctx.logger.warn).toHaveBeenCalledWith(expect.stringContaining('MLflow setup failed'));
			expect(mockedToolsAgentExecute).toHaveBeenCalled();
		});

		it('should log when MLflow tracing is enabled', async () => {
			const ctx = createMockContext();
			mockedSetupMlflow.mockResolvedValue(new MlflowCallbackHandler());

			await agent.execute.call(ctx);

			expect(ctx.logger.info).toHaveBeenCalledWith('MLflow tracing enabled for this execution');
		});

		it('should not log MLflow enabled when handler is undefined', async () => {
			const ctx = createMockContext();
			mockedSetupMlflow.mockResolvedValue(undefined);

			await agent.execute.call(ctx);

			expect(ctx.logger.info).not.toHaveBeenCalledWith('MLflow tracing enabled for this execution');
		});

		it('should return the result from toolsAgentExecute', async () => {
			const ctx = createMockContext();
			const expected = [[{ json: { output: 'hello' } }]];
			mockedToolsAgentExecute.mockResolvedValue(expected as any);
			mockedSetupMlflow.mockResolvedValue(undefined);

			const result = await agent.execute.call(ctx);

			expect(result).toBe(expected);
		});
	});
});
