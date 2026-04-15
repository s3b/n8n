import * as mlflow from 'mlflow-tracing';
import type { IExecuteFunctions, ISupplyDataFunctions } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { MlflowCallbackHandler } from './MlflowCallbackHandler';
import { validateDatabricksHost } from './databricks-utils';

/**
 * Sets up MLflow experiment tracking for a Databricks AI Agent execution.
 * Creates or retrieves the experiment and initializes the MLflow tracing client.
 *
 * @returns The MlflowCallbackHandler to inject into the agent's callback chain,
 *          or undefined if MLflow is disabled or initialization fails.
 */
export async function setupMlflowTracing(
	ctx: IExecuteFunctions | ISupplyDataFunctions,
): Promise<MlflowCallbackHandler | undefined> {
	const enableMlflow = ctx.getNodeParameter('enableMlflow', 0, false) as boolean;

	if (!enableMlflow) {
		return undefined;
	}

	const credentials = (await ctx.getCredentials('databricksApi')) as {
		host: string;
		token: string;
	};

	if (!credentials?.host || !credentials?.token) {
		throw new NodeOperationError(
			ctx.getNode(),
			'Databricks credentials are required when MLflow logging is enabled. Please configure valid host and token.',
		);
	}

	const host = validateDatabricksHost(credentials.host);

	// Determine experiment name based on workflow
	const workflowId = 'getWorkflow' in ctx ? ctx.getWorkflow().id : 'unknown';
	const experimentName = `/Shared/n8n-workflows-${workflowId}`;

	// Create or get the MLflow experiment
	try {
		await ctx.helpers.httpRequestWithAuthentication.call(ctx, 'databricksApi', {
			method: 'GET',
			url: `${host}/api/2.0/mlflow/experiments/get-by-name`,
			qs: { experiment_name: experimentName },
			headers: { Accept: 'application/json' },
			json: true,
		});
	} catch {
		// Experiment doesn't exist — try to create it
		try {
			await ctx.helpers.httpRequestWithAuthentication.call(ctx, 'databricksApi', {
				method: 'POST',
				url: `${host}/api/2.0/mlflow/experiments/create`,
				headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
				body: { name: experimentName },
				json: true,
			});
			ctx.logger.info(`Created MLflow experiment: ${experimentName}`);
		} catch (createError) {
			ctx.logger.warn(
				`Could not create MLflow experiment "${experimentName}": ${createError instanceof Error ? createError.message : String(createError)}`,
			);
		}
	}

	// Initialize MLflow tracing client with save/restore of env vars to prevent
	// credential leakage across concurrent workflow executions
	const prevHost = process.env.DATABRICKS_HOST;
	const prevToken = process.env.DATABRICKS_TOKEN;
	const prevUri = process.env.MLFLOW_TRACKING_URI;

	try {
		process.env.DATABRICKS_HOST = host;
		process.env.DATABRICKS_TOKEN = credentials.token;
		process.env.MLFLOW_TRACKING_URI = 'databricks';

		mlflow.init({
			trackingUri: 'databricks',
			experimentId: experimentName,
		});
	} catch (error) {
		ctx.logger.warn(
			`MLflow initialization failed, continuing without tracing: ${error instanceof Error ? error.message : String(error)}`,
		);
		return undefined;
	} finally {
		// Restore previous env vars to prevent credential leakage
		if (prevHost !== undefined) {
			process.env.DATABRICKS_HOST = prevHost;
		} else {
			delete process.env.DATABRICKS_HOST;
		}
		if (prevToken !== undefined) {
			process.env.DATABRICKS_TOKEN = prevToken;
		} else {
			delete process.env.DATABRICKS_TOKEN;
		}
		if (prevUri !== undefined) {
			process.env.MLFLOW_TRACKING_URI = prevUri;
		} else {
			delete process.env.MLFLOW_TRACKING_URI;
		}
	}

	return new MlflowCallbackHandler(ctx.logger);
}
