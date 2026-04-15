import * as mlflow from 'mlflow-tracing';

import { MlflowCallbackHandler } from './MlflowCallbackHandler';

// Mock mlflow-tracing
jest.mock('mlflow-tracing', () => ({
	startSpan: jest.fn(),
	SpanType: {
		AGENT: 'AGENT',
		CHAT_MODEL: 'CHAT_MODEL',
		TOOL: 'TOOL',
		RETRIEVER: 'RETRIEVER',
	},
	SpanStatusCode: {
		ERROR: 'STATUS_CODE_ERROR',
	},
}));

const mockStartSpan = mlflow.startSpan as jest.MockedFunction<typeof mlflow.startSpan>;

describe('MlflowCallbackHandler', () => {
	let handler: MlflowCallbackHandler;

	const createMockSpan = () => ({
		setInputs: jest.fn(),
		setOutputs: jest.fn(),
		setAttribute: jest.fn(),
		setStatus: jest.fn(),
		end: jest.fn(),
		traceId: 'test-trace-id',
	});

	beforeEach(() => {
		handler = new MlflowCallbackHandler();
		jest.clearAllMocks();
	});

	describe('handleChainStart', () => {
		it('should create an AGENT span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleChainStart(
				{ id: ['test-chain'], lc: 1, type: 'not_implemented' },
				{ input: 'hello' },
				'run-1',
			);

			expect(mockStartSpan).toHaveBeenCalledWith({
				name: 'test-chain',
				spanType: 'AGENT',
			});
			expect(mockSpan.setInputs).toHaveBeenCalledWith({ input: 'hello' });
		});

		it('should skip internal chain names', async () => {
			await handler.handleChainStart(
				{ id: ['RunnableLambda'], lc: 1, type: 'not_implemented' },
				{},
				'run-1',
			);

			expect(mockStartSpan).not.toHaveBeenCalled();
		});

		it('should skip RunnableSequence', async () => {
			await handler.handleChainStart(
				{ id: ['RunnableSequence'], lc: 1, type: 'not_implemented' },
				{},
				'run-2',
			);

			expect(mockStartSpan).not.toHaveBeenCalled();
		});

		it('should use serialized name if available', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleChainStart(
				{ name: 'MyAgent', id: ['fallback'], lc: 1, type: 'not_implemented' },
				{},
				'run-3',
			);

			expect(mockStartSpan).toHaveBeenCalledWith(expect.objectContaining({ name: 'MyAgent' }));
		});
	});

	describe('handleChainEnd', () => {
		it('should end the span with outputs', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleChainStart(
				{ id: ['chain'], lc: 1, type: 'not_implemented' },
				{},
				'run-1',
			);
			await handler.handleChainEnd({ output: 'result' }, 'run-1');

			expect(mockSpan.setOutputs).toHaveBeenCalledWith({ output: 'result' });
			expect(mockSpan.end).toHaveBeenCalled();
		});

		it('should be a no-op for unknown runId', async () => {
			await handler.handleChainEnd({ output: 'result' }, 'unknown-run');
			// Should not throw
		});
	});

	describe('handleChainError', () => {
		it('should set error status and end span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleChainStart(
				{ id: ['chain'], lc: 1, type: 'not_implemented' },
				{},
				'run-1',
			);
			await handler.handleChainError(new Error('test error'), 'run-1');

			expect(mockSpan.setStatus).toHaveBeenCalledWith('STATUS_CODE_ERROR');
			expect(mockSpan.end).toHaveBeenCalled();
		});
	});

	describe('handleChatModelStart', () => {
		it('should create a CHAT_MODEL span with messages', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			const messages = [[{ _getType: () => 'human', content: 'hello' }]] as any;

			await handler.handleChatModelStart(
				{ id: ['ChatModel'], lc: 1, type: 'not_implemented' },
				messages,
				'run-1',
				undefined,
				{
					invocation_params: {
						model: 'databricks-claude-sonnet-4-6',
						temperature: 0.7,
						max_tokens: 1000,
					},
				},
			);

			expect(mockStartSpan).toHaveBeenCalledWith({
				name: 'ChatModel',
				spanType: 'CHAT_MODEL',
			});
			expect(mockSpan.setInputs).toHaveBeenCalledWith({
				messages: [{ role: 'human', content: 'hello' }],
			});
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('model', 'databricks-claude-sonnet-4-6');
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('temperature', '0.7');
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('max_tokens', '1000');
		});
	});

	describe('handleLLMStart', () => {
		it('should create a CHAT_MODEL span with prompts', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleLLMStart(
				{ id: ['LLM'], lc: 1, type: 'not_implemented' },
				['prompt 1', 'prompt 2'],
				'run-1',
			);

			expect(mockStartSpan).toHaveBeenCalledWith({
				name: 'LLM',
				spanType: 'CHAT_MODEL',
			});
			expect(mockSpan.setInputs).toHaveBeenCalledWith({
				prompts: ['prompt 1', 'prompt 2'],
			});
		});
	});

	describe('handleLLMEnd', () => {
		it('should set outputs and token usage', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleLLMStart({ id: ['LLM'], lc: 1, type: 'not_implemented' }, [], 'run-1');

			await handler.handleLLMEnd(
				{
					generations: [
						[
							{
								text: 'Hello!',
								message: {
									response_metadata: { model_name: 'test-model' },
									usage_metadata: {
										input_tokens: 10,
										output_tokens: 5,
										total_tokens: 15,
									},
								},
							},
						],
					],
				} as any,
				'run-1',
			);

			expect(mockSpan.setOutputs).toHaveBeenCalledWith({
				messages: [{ content: 'Hello!' }],
			});
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('input_tokens', '10');
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('output_tokens', '5');
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('total_tokens', '15');
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('model_name', 'test-model');
			expect(mockSpan.end).toHaveBeenCalled();
		});
	});

	describe('handleLLMError', () => {
		it('should set error status and end span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleLLMStart({ id: ['LLM'], lc: 1, type: 'not_implemented' }, [], 'run-1');
			await handler.handleLLMError(new Error('LLM failed'), 'run-1');

			expect(mockSpan.setStatus).toHaveBeenCalledWith('STATUS_CODE_ERROR');
			expect(mockSpan.end).toHaveBeenCalled();
		});
	});

	describe('handleToolStart', () => {
		it('should create a TOOL span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleToolStart(
				{ name: 'calculator', id: ['tool'], lc: 1, type: 'not_implemented' },
				'2 + 2',
				'run-1',
			);

			expect(mockStartSpan).toHaveBeenCalledWith({
				name: 'calculator',
				spanType: 'TOOL',
			});
			expect(mockSpan.setInputs).toHaveBeenCalledWith({ input: '2 + 2' });
		});
	});

	describe('handleToolEnd', () => {
		it('should set output and end span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleToolStart(
				{ id: ['tool'], lc: 1, type: 'not_implemented' },
				'input',
				'run-1',
			);
			await handler.handleToolEnd('4', 'run-1');

			expect(mockSpan.setOutputs).toHaveBeenCalledWith({ output: '4' });
			expect(mockSpan.end).toHaveBeenCalled();
		});
	});

	describe('handleToolError', () => {
		it('should set error status and end span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleToolStart(
				{ id: ['tool'], lc: 1, type: 'not_implemented' },
				'input',
				'run-1',
			);
			await handler.handleToolError(new Error('tool failed'), 'run-1');

			expect(mockSpan.setStatus).toHaveBeenCalledWith('STATUS_CODE_ERROR');
			expect(mockSpan.end).toHaveBeenCalled();
		});
	});

	describe('handleAgentAction', () => {
		it('should set agent action attributes', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleChainStart(
				{ id: ['agent'], lc: 1, type: 'not_implemented' },
				{},
				'run-1',
			);
			await handler.handleAgentAction(
				{ tool: 'search', toolInput: { query: 'test' }, log: '' },
				'run-1',
			);

			expect(mockSpan.setAttribute).toHaveBeenCalledWith('agent_action_tool', 'search');
			expect(mockSpan.setAttribute).toHaveBeenCalledWith('agent_action_input', '{"query":"test"}');
		});
	});

	describe('handleAgentEnd', () => {
		it('should set return values as outputs', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleChainStart(
				{ id: ['agent'], lc: 1, type: 'not_implemented' },
				{},
				'run-1',
			);
			await handler.handleAgentEnd({ returnValues: { output: 'final answer' }, log: '' }, 'run-1');

			expect(mockSpan.setOutputs).toHaveBeenCalledWith({ output: 'final answer' });
		});
	});

	describe('handleRetrieverStart', () => {
		it('should create a RETRIEVER span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleRetrieverStart(
				{ id: ['retriever'], lc: 1, type: 'not_implemented' },
				'search query',
				'run-1',
			);

			expect(mockStartSpan).toHaveBeenCalledWith({
				name: 'retriever',
				spanType: 'RETRIEVER',
			});
			expect(mockSpan.setInputs).toHaveBeenCalledWith({ query: 'search query' });
		});
	});

	describe('handleRetrieverEnd', () => {
		it('should set documents as output and end span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleRetrieverStart(
				{ id: ['retriever'], lc: 1, type: 'not_implemented' },
				'query',
				'run-1',
			);
			await handler.handleRetrieverEnd(
				[{ pageContent: 'doc content', metadata: { source: 'test' } }],
				'run-1',
			);

			expect(mockSpan.setOutputs).toHaveBeenCalledWith({
				documents: [{ content: 'doc content', metadata: { source: 'test' } }],
			});
			expect(mockSpan.end).toHaveBeenCalled();
		});
	});

	describe('handleRetrieverError', () => {
		it('should set error status and end span', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			await handler.handleRetrieverStart(
				{ id: ['retriever'], lc: 1, type: 'not_implemented' },
				'query',
				'run-1',
			);
			await handler.handleRetrieverError(new Error('retriever failed'), 'run-1');

			expect(mockSpan.setStatus).toHaveBeenCalledWith('STATUS_CODE_ERROR');
			expect(mockSpan.end).toHaveBeenCalled();
		});
	});

	describe('error resilience', () => {
		it('should not throw when mlflow.startSpan throws', async () => {
			mockStartSpan.mockImplementation(() => {
				throw new Error('mlflow unavailable');
			});

			// Should not throw
			await handler.handleChainStart(
				{ id: ['chain'], lc: 1, type: 'not_implemented' },
				{},
				'run-1',
			);
		});

		it('should respect maxMapSize limit', async () => {
			const mockSpan = createMockSpan();
			mockStartSpan.mockReturnValue(mockSpan as any);

			// Fill up the map to the limit
			for (let i = 0; i < 1000; i++) {
				await handler.handleChainStart(
					{ id: ['chain'], lc: 1, type: 'not_implemented' },
					{},
					`run-${i}`,
				);
			}

			mockStartSpan.mockClear();

			// This should be silently skipped
			await handler.handleChainStart(
				{ id: ['chain'], lc: 1, type: 'not_implemented' },
				{},
				'run-overflow',
			);

			expect(mockStartSpan).not.toHaveBeenCalled();
		});
	});
});
