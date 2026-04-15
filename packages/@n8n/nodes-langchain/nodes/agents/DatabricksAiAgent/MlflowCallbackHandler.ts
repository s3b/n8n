import type { Serialized } from '@langchain/core/load/serializable';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { LLMResult, Generation } from '@langchain/core/outputs';
import type { ChainValues } from '@langchain/core/utils/types';
import type { AgentAction, AgentFinish } from '@langchain/core/agents';
import * as mlflow from 'mlflow-tracing';

type ErrorLike = Error | { message: string; stack?: string; [key: string]: unknown };

interface GenerationWithMetadata extends Generation {
	message?: {
		response_metadata?: { model_name?: string };
		usage_metadata?: UsageMetadata;
	};
}

interface LoggerLike {
	warn(message: string): void;
	error(message: string): void;
}

type MlflowSpan = ReturnType<typeof mlflow.startSpan>;

/**
 * MLflow tracing callback handler for LangChain agents.
 * Logs AGENT, CHAT_MODEL, TOOL, and RETRIEVER spans to Databricks MLflow.
 */
export class MlflowCallbackHandler extends BaseCallbackHandler {
	name = 'MlflowCallbackHandler';

	private readonly runMap = new Map<string, MlflowSpan>();

	private readonly maxMapSize = 1000;

	private readonly logger?: LoggerLike;

	constructor(logger?: LoggerLike) {
		super();
		this.logger = logger;
	}

	private readonly internalChainNames = new Set([
		'RunnableLambda',
		'RunnableMap',
		'RunnableParallel',
		'RunnablePassthrough',
		'RunnableAssign',
		'RunnableSequence',
		'RunnablePick',
		'ChatPromptTemplate',
		'PromptTemplate',
	]);

	private logError(context: string, error: unknown): void {
		const msg = error instanceof Error ? error.message : String(error);
		if (this.logger) {
			this.logger.warn(`[MlflowCallbackHandler] Error in ${context}: ${msg}`);
		}
	}

	private cleanupRun(runId: string): void {
		this.runMap.delete(runId);
	}

	// --- Chain / Agent Events ---

	async handleChainStart(
		serialized: Serialized,
		inputs: ChainValues,
		runId: string,
		_parentRunId?: string,
	): Promise<void> {
		try {
			const name = serialized?.name ?? serialized?.id?.at(-1) ?? 'UnknownChain';
			if (this.internalChainNames.has(name)) return;
			if (this.runMap.size >= this.maxMapSize) return;

			const span = mlflow.startSpan({ name, spanType: mlflow.SpanType.AGENT });
			if (inputs) {
				try {
					const messages = inputs.input ?? inputs.messages ?? inputs;
					span.setInputs(typeof messages === 'string' ? { input: messages } : messages);
				} catch {
					// ignore serialization errors
				}
			}
			this.runMap.set(runId, span);
		} catch (error) {
			this.logError('handleChainStart', error);
		}
	}

	async handleChainEnd(outputs: ChainValues, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			try {
				span.setOutputs(outputs);
			} catch {
				// ignore
			}
			span.end();
			this.cleanupRun(runId);
		} catch (error) {
			this.logError('handleChainEnd', error);
		}
	}

	async handleChainError(_error: ErrorLike, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setStatus(mlflow.SpanStatusCode.ERROR);
			span.end();
			this.cleanupRun(runId);
		} catch (err) {
			this.logError('handleChainError', err);
		}
	}

	// --- LLM Events ---

	async handleLLMStart(serialized: Serialized, prompts: string[], runId: string): Promise<void> {
		try {
			if (this.runMap.size >= this.maxMapSize) return;
			const name = serialized?.name ?? serialized?.id?.at(-1) ?? 'LLM';
			const span = mlflow.startSpan({ name, spanType: mlflow.SpanType.CHAT_MODEL });
			span.setInputs({ prompts });
			this.runMap.set(runId, span);
		} catch (error) {
			this.logError('handleLLMStart', error);
		}
	}

	async handleChatModelStart(
		serialized: Serialized,
		messages: BaseMessage[][],
		runId: string,
		_parentRunId?: string,
		extraParams?: Record<string, unknown>,
	): Promise<void> {
		try {
			if (this.runMap.size >= this.maxMapSize) return;
			const name = serialized?.name ?? serialized?.id?.at(-1) ?? 'ChatModel';
			const span = mlflow.startSpan({ name, spanType: mlflow.SpanType.CHAT_MODEL });

			const formattedMessages = messages.flat().map((m) => ({
				role: m._getType(),
				content: m.content,
			}));
			span.setInputs({ messages: formattedMessages });

			if (extraParams?.invocation_params) {
				const params = extraParams.invocation_params as Record<string, unknown>;
				span.setAttribute('model', String(params.model ?? params.model_name ?? ''));
				span.setAttribute('temperature', String(params.temperature ?? ''));
				span.setAttribute('max_tokens', String(params.max_tokens ?? ''));
			}
			this.runMap.set(runId, span);
		} catch (error) {
			this.logError('handleChatModelStart', error);
		}
	}

	async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;

			const generations = output.generations?.flat() ?? [];
			span.setOutputs({ messages: generations.map((gen) => ({ content: gen.text })) });

			const firstGen = generations[0] as GenerationWithMetadata | undefined;
			if (firstGen?.message?.usage_metadata) {
				const usage = firstGen.message.usage_metadata;
				span.setAttribute('input_tokens', String(usage.input_tokens ?? 0));
				span.setAttribute('output_tokens', String(usage.output_tokens ?? 0));
				span.setAttribute(
					'total_tokens',
					String(usage.total_tokens ?? (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0)),
				);
			}
			if (firstGen?.message?.response_metadata?.model_name) {
				span.setAttribute('model_name', firstGen.message.response_metadata.model_name);
			}

			span.end();
			this.cleanupRun(runId);
		} catch (error) {
			this.logError('handleLLMEnd', error);
		}
	}

	async handleLLMError(_error: ErrorLike, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setStatus(mlflow.SpanStatusCode.ERROR);
			span.end();
			this.cleanupRun(runId);
		} catch (err) {
			this.logError('handleLLMError', err);
		}
	}

	// --- Tool Events ---

	async handleToolStart(tool: Serialized, input: string, runId: string): Promise<void> {
		try {
			if (this.runMap.size >= this.maxMapSize) return;
			const name = tool?.name ?? tool?.id?.at(-1) ?? 'Tool';
			const span = mlflow.startSpan({ name, spanType: mlflow.SpanType.TOOL });
			span.setInputs({ input });
			this.runMap.set(runId, span);
		} catch (error) {
			this.logError('handleToolStart', error);
		}
	}

	async handleToolEnd(output: string, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setOutputs({ output });
			span.end();
			this.cleanupRun(runId);
		} catch (error) {
			this.logError('handleToolEnd', error);
		}
	}

	async handleToolError(_error: ErrorLike, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setStatus(mlflow.SpanStatusCode.ERROR);
			span.end();
			this.cleanupRun(runId);
		} catch (err) {
			this.logError('handleToolError', err);
		}
	}

	// --- Agent Events ---

	async handleAgentAction(action: AgentAction, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setAttribute('agent_action_tool', action.tool);
			span.setAttribute('agent_action_input', JSON.stringify(action.toolInput));
		} catch (error) {
			this.logError('handleAgentAction', error);
		}
	}

	async handleAgentEnd(action: AgentFinish, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setOutputs(action.returnValues);
		} catch (error) {
			this.logError('handleAgentEnd', error);
		}
	}

	// --- Retriever Events ---

	async handleRetrieverStart(retriever: Serialized, query: string, runId: string): Promise<void> {
		try {
			if (this.runMap.size >= this.maxMapSize) return;
			const name = retriever?.name ?? retriever?.id?.at(-1) ?? 'Retriever';
			const span = mlflow.startSpan({ name, spanType: mlflow.SpanType.RETRIEVER });
			span.setInputs({ query });
			this.runMap.set(runId, span);
		} catch (error) {
			this.logError('handleRetrieverStart', error);
		}
	}

	async handleRetrieverEnd(
		documents: Array<{ pageContent: string; metadata: Record<string, unknown> }>,
		runId: string,
	): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setOutputs({
				documents: documents.map((d) => ({
					content: d.pageContent.substring(0, 500),
					metadata: d.metadata,
				})),
			});
			span.end();
			this.cleanupRun(runId);
		} catch (error) {
			this.logError('handleRetrieverEnd', error);
		}
	}

	async handleRetrieverError(_error: ErrorLike, runId: string): Promise<void> {
		try {
			const span = this.runMap.get(runId);
			if (!span) return;
			span.setStatus(mlflow.SpanStatusCode.ERROR);
			span.end();
			this.cleanupRun(runId);
		} catch (err) {
			this.logError('handleRetrieverError', err);
		}
	}
}
