import type { IExecuteFunctions } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';
import { mock } from 'jest-mock-extended';

jest.mock('mlflow-tracing', () => ({
	init: jest.fn(),
	startSpan: jest.fn(),
	SpanType: { AGENT: 'AGENT' },
	SpanStatusCode: { ERROR: 'ERROR' },
}));

import * as mlflow from 'mlflow-tracing';

import { setupMlflowTracing } from './mlflow-utils';
import { MlflowCallbackHandler } from './MlflowCallbackHandler';

const mockedMlflowInit = mlflow.init as jest.MockedFunction<typeof mlflow.init>;

describe('setupMlflowTracing', () => {
	const createMockContext = (params: Record<string, unknown> = {}) => {
		const ctx = mock<IExecuteFunctions>();
		ctx.getNodeParameter = jest
			.fn()
			.mockImplementation((name: string, _idx: number, def: unknown) => {
				return params[name] ?? def;
			});
		ctx.getNode = jest.fn().mockReturnValue({ name: 'TestAgent' });
		ctx.getWorkflow = jest.fn().mockReturnValue({ id: 'wf-123', name: 'Test Workflow' });
		ctx.logger = {
			debug: jest.fn(),
			info: jest.fn(),
			warn: jest.fn(),
			error: jest.fn(),
			verbose: jest.fn(),
		} as any;
		ctx.helpers = {
			httpRequestWithAuthentication: {
				call: jest.fn().mockResolvedValue({ experiment: { experiment_id: 'exp-1' } }),
			},
		} as any;
		return ctx;
	};

	beforeEach(() => {
		jest.clearAllMocks();
		delete process.env.DATABRICKS_HOST;
		delete process.env.DATABRICKS_TOKEN;
		delete process.env.MLFLOW_TRACKING_URI;
	});

	it('should return undefined when MLflow is disabled', async () => {
		const ctx = createMockContext({ enableMlflow: false });

		const result = await setupMlflowTracing(ctx);

		expect(result).toBeUndefined();
		expect(mockedMlflowInit).not.toHaveBeenCalled();
	});

	it('should return MlflowCallbackHandler when MLflow is enabled', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com',
			token: 'dapi-test-token',
		});

		const result = await setupMlflowTracing(ctx);

		expect(result).toBeInstanceOf(MlflowCallbackHandler);
	});

	it('should throw when credentials are missing', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: '',
			token: '',
		});

		await expect(setupMlflowTracing(ctx)).rejects.toThrow(NodeOperationError);
	});

	it('should throw when host does not start with https://', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'http://dbc-test.cloud.databricks.com',
			token: 'dapi-test',
		});

		await expect(setupMlflowTracing(ctx)).rejects.toThrow('https://');
	});

	it('should strip trailing slash from host', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com/',
			token: 'dapi-test',
		});

		await setupMlflowTracing(ctx);

		// Env vars are cleaned up in finally block — verify init was called with correct host
		expect(mockedMlflowInit).toHaveBeenCalledWith(
			expect.objectContaining({ trackingUri: 'databricks' }),
		);
	});

	it('should clean up environment variables after init (security)', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com',
			token: 'dapi-test-token',
		});

		await setupMlflowTracing(ctx);

		// Env vars should be cleaned up after init to prevent credential leakage
		expect(process.env.DATABRICKS_HOST).toBeUndefined();
		expect(process.env.DATABRICKS_TOKEN).toBeUndefined();
		expect(process.env.MLFLOW_TRACKING_URI).toBeUndefined();
	});

	it('should call mlflow.init with trackingUri and experimentId', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com',
			token: 'dapi-test',
		});

		await setupMlflowTracing(ctx);

		expect(mockedMlflowInit).toHaveBeenCalledWith({
			trackingUri: 'databricks',
			experimentId: '/Shared/n8n-workflows-wf-123',
		});
	});

	it('should try to get experiment by name via API', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com',
			token: 'dapi-test',
		});

		await setupMlflowTracing(ctx);

		expect(ctx.helpers.httpRequestWithAuthentication.call).toHaveBeenCalledWith(
			ctx,
			'databricksApi',
			expect.objectContaining({
				method: 'GET',
				url: 'https://dbc-test.cloud.databricks.com/api/2.0/mlflow/experiments/get-by-name',
				qs: { experiment_name: '/Shared/n8n-workflows-wf-123' },
			}),
		);
	});

	it('should try to create experiment if get-by-name fails', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com',
			token: 'dapi-test',
		});

		// First call (get-by-name) fails, second call (create) succeeds
		(ctx.helpers.httpRequestWithAuthentication.call as jest.Mock)
			.mockRejectedValueOnce(new Error('Not found'))
			.mockResolvedValueOnce({ experiment_id: 'new-exp' });

		await setupMlflowTracing(ctx);

		expect(ctx.helpers.httpRequestWithAuthentication.call).toHaveBeenCalledTimes(2);
		expect(ctx.helpers.httpRequestWithAuthentication.call).toHaveBeenLastCalledWith(
			ctx,
			'databricksApi',
			expect.objectContaining({
				method: 'POST',
				url: 'https://dbc-test.cloud.databricks.com/api/2.0/mlflow/experiments/create',
				body: { name: '/Shared/n8n-workflows-wf-123' },
			}),
		);
	});

	it('should warn but not throw if experiment creation fails', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com',
			token: 'dapi-test',
		});

		(ctx.helpers.httpRequestWithAuthentication.call as jest.Mock)
			.mockRejectedValueOnce(new Error('Not found'))
			.mockRejectedValueOnce(new Error('Permission denied'));

		const result = await setupMlflowTracing(ctx);

		expect(result).toBeInstanceOf(MlflowCallbackHandler);
		expect(ctx.logger.warn).toHaveBeenCalledWith(expect.stringContaining('Permission denied'));
	});

	it('should return undefined and warn if mlflow.init fails', async () => {
		const ctx = createMockContext({ enableMlflow: true });
		ctx.getCredentials = jest.fn().mockResolvedValue({
			host: 'https://dbc-test.cloud.databricks.com',
			token: 'dapi-test',
		});
		mockedMlflowInit.mockImplementation(() => {
			throw new Error('init failed');
		});

		const result = await setupMlflowTracing(ctx);

		expect(result).toBeUndefined();
		expect(ctx.logger.warn).toHaveBeenCalledWith(expect.stringContaining('init failed'));
	});
});
