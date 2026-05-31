import type { EngineResponse, IExecuteFunctions, INodeExecutionData } from 'n8n-workflow';

import { toolsAgentExecute } from '../../Agent/agents/ToolsAgent/V3/execute';
import { DatabricksAiAgent } from '../DatabricksAiAgent.node';

vi.mock('../../Agent/agents/ToolsAgent/V3/execute', () => ({
	toolsAgentExecute: vi.fn(),
}));

const mockedExecute = vi.mocked(toolsAgentExecute);

describe('DatabricksAiAgent', () => {
	let node: DatabricksAiAgent;

	beforeEach(() => {
		node = new DatabricksAiAgent();
		vi.clearAllMocks();
	});

	describe('description', () => {
		it('should have correct node metadata', () => {
			expect(node.description.name).toBe('databricksAiAgent');
			expect(node.description.displayName).toBe('Databricks AI Agent');
			expect(node.description.outputs).toEqual(['main']);
		});

		it('should be categorised as an AI Agent root node', () => {
			expect(node.description.codex?.subcategories?.AI).toEqual(['Agents', 'Root Nodes']);
		});

		it('should require a language model input', () => {
			expect(node.description.builderHint?.inputs?.ai_languageModel).toEqual({ required: true });
		});

		it('should expose the shared tools-agent options', () => {
			const options = node.description.properties.find((p) => p.name === 'options');
			expect(options?.type).toBe('collection');
		});
	});

	describe('execute', () => {
		it('should delegate to the V3 tools-agent executor and forward the response', async () => {
			const expected: INodeExecutionData[][] = [[{ json: { output: 'hello' } }]];
			mockedExecute.mockResolvedValue(expected);

			const ctx = { getNode: vi.fn() } as unknown as IExecuteFunctions;
			const engineResponse = { foo: 'bar' } as unknown as EngineResponse<never>;

			const result = await node.execute.call(ctx, engineResponse);

			expect(mockedExecute).toHaveBeenCalledTimes(1);
			expect(mockedExecute).toHaveBeenCalledWith(engineResponse);
			expect(mockedExecute.mock.contexts[0]).toBe(ctx);
			expect(result).toBe(expected);
		});
	});
});
