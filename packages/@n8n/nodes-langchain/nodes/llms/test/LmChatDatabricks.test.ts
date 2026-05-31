import { ChatOpenAI } from '@langchain/openai';
import type * as AiUtilities from '@n8n/ai-utilities';
import { createMockExecuteFunction } from 'n8n-nodes-base/test/nodes/Helpers';
import type { INode, ISupplyDataFunctions } from 'n8n-workflow';
import { lookup } from 'node:dns/promises';
import type { Mock } from 'vitest';

import {
	validateDatabricksHost,
	validateResourceName,
} from '../../vendors/Databricks/databricks-utils';
import { LmChatDatabricks } from '../LmChatDatabricks/LmChatDatabricks.node';

vi.mock('@langchain/openai');

vi.mock('@n8n/ai-utilities', async () => {
	const actual = await vi.importActual<typeof AiUtilities>('@n8n/ai-utilities');
	return {
		...actual,
		N8nLlmTracing: vi.fn(),
		makeN8nLlmFailedAttemptHandler: vi.fn().mockReturnValue(vi.fn()),
		getConnectionHintNoticeField: vi.fn().mockReturnValue({}),
		getProxyAgent: vi.fn().mockReturnValue({}),
	};
});

vi.mock('node:dns/promises', () => ({
	lookup: vi.fn(),
}));

const MockedChatOpenAI = vi.mocked(ChatOpenAI);
// `dns.lookup` is overloaded; cast so the `{ all: true }` array form is mockable.
const mockedLookup = vi.mocked(lookup) as unknown as Mock;

describe('LmChatDatabricks', () => {
	let node: LmChatDatabricks;

	const mockNode: INode = {
		id: '1',
		name: 'Databricks Chat Model',
		typeVersion: 1,
		type: '@n8n/n8n-nodes-langchain.lmChatDatabricks',
		position: [0, 0],
		parameters: {},
	};

	const databricksCredentials = {
		host: 'https://dbc-test.cloud.databricks.com',
		token: 'dapi-test-token-123',
	};

	const setupMockContext = (nodeOverrides: Partial<INode> = {}) => {
		const nodeConfig = { ...mockNode, ...nodeOverrides };
		const mockContext = createMockExecuteFunction<ISupplyDataFunctions>({}, nodeConfig);

		mockContext.getCredentials = vi.fn().mockResolvedValue(databricksCredentials);
		mockContext.getNode = vi.fn().mockReturnValue(nodeConfig);
		mockContext.getNodeParameter = vi.fn();
		mockContext.logger = {
			debug: vi.fn(),
			info: vi.fn(),
			warn: vi.fn(),
			error: vi.fn(),
		};
		return mockContext;
	};

	beforeEach(() => {
		node = new LmChatDatabricks();
		vi.clearAllMocks();
		// Default: hostnames resolve to a public, unicast address.
		mockedLookup.mockResolvedValue([{ address: '52.1.2.3', family: 4 }]);
	});

	describe('description', () => {
		it('should have correct node metadata', () => {
			expect(node.description.name).toBe('lmChatDatabricks');
			expect(node.description.displayName).toBe('Databricks Chat Model');
			expect(node.description.credentials).toEqual([{ name: 'databricksApi', required: true }]);
			expect(node.description.outputs).toEqual(['ai_languageModel']);
		});

		it('should have requestDefaults with credential-based baseURL', () => {
			expect(node.description.requestDefaults).toEqual(
				expect.objectContaining({
					baseURL: '={{$credentials.host}}',
				}),
			);
		});

		it('should load model options via the getModels method', () => {
			const modelProp = node.description.properties.find((p) => p.name === 'model');
			expect(modelProp?.type).toBe('options');
			expect(modelProp?.typeOptions?.loadOptionsMethod).toBe('getModels');
		});
	});

	describe('supplyData', () => {
		it('should create ChatOpenAI with correct Databricks configuration', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'databricks-claude-sonnet-4-6';
				if (paramName === 'options') return {};
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					apiKey: 'dapi-test-token-123',
					model: 'databricks-claude-sonnet-4-6',
					configuration: expect.objectContaining({
						baseURL: 'https://dbc-test.cloud.databricks.com/serving-endpoints',
					}),
				}),
			);
		});

		it('should strip trailing slash from host', async () => {
			const ctx = setupMockContext();
			ctx.getCredentials = vi.fn().mockResolvedValue({
				host: 'https://dbc-test.cloud.databricks.com/',
				token: 'dapi-test',
			});
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-endpoint';
				if (paramName === 'options') return {};
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					configuration: expect.objectContaining({
						baseURL: 'https://dbc-test.cloud.databricks.com/serving-endpoints',
					}),
				}),
			);
		});

		it('should pass options through to ChatOpenAI', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options')
					return {
						temperature: 0.5,
						maxTokens: 1000,
						topP: 0.9,
						maxRetries: 3,
						timeout: 30000,
					};
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					temperature: 0.5,
					maxTokens: 1000,
					topP: 0.9,
					maxRetries: 3,
					timeout: 30000,
				}),
			);
		});

		it('should set response_format in modelKwargs when JSON format requested', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return { responseFormat: 'json_object' };
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					modelKwargs: { response_format: { type: 'json_object' } },
				}),
			);
		});

		it('should not set modelKwargs when no responseFormat', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return {};
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					modelKwargs: undefined,
				}),
			);
		});

		it('should use default timeout of 60000', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return {};
				return undefined;
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					timeout: 60000,
				}),
			);
		});

		it('should return model as response', async () => {
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return {};
				return undefined;
			});

			const result = await node.supplyData.call(ctx, 0);

			expect(result.response).toBeDefined();
			expect(result.response).toBeInstanceOf(ChatOpenAI);
		});

		it('should reject a host that resolves to a private address', async () => {
			mockedLookup.mockResolvedValue([{ address: '169.254.169.254', family: 4 }]);
			const ctx = setupMockContext();
			ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
				if (paramName === 'model') return 'test-model';
				if (paramName === 'options') return {};
				return undefined;
			});

			await expect(node.supplyData.call(ctx, 0)).rejects.toThrow(
				'must be a public Databricks workspace URL',
			);
			expect(MockedChatOpenAI).not.toHaveBeenCalled();
		});
	});
});

describe('databricks-utils', () => {
	const mockedLookup = vi.mocked(lookup) as unknown as Mock;

	beforeEach(() => {
		vi.clearAllMocks();
		mockedLookup.mockResolvedValue([{ address: '52.1.2.3', family: 4 }]);
	});

	describe('validateDatabricksHost', () => {
		it('should strip trailing slashes from a valid public host', async () => {
			await expect(validateDatabricksHost('https://example.databricks.com///')).resolves.toBe(
				'https://example.databricks.com',
			);
		});

		it('should reject a non-HTTPS host', async () => {
			await expect(validateDatabricksHost('http://example.databricks.com')).rejects.toThrow(
				'must use HTTPS',
			);
		});

		it('should reject an invalid URL', async () => {
			await expect(validateDatabricksHost('not a url')).rejects.toThrow('must be a valid URL');
		});

		it.each([
			['loopback', '127.0.0.1'],
			['private 10/8', '10.0.0.5'],
			['link-local metadata', '169.254.169.254'],
			['carrier-grade NAT metadata', '100.100.100.200'],
			['IPv6 loopback', '::1'],
		])('should reject a host resolving to %s', async (_label, address) => {
			mockedLookup.mockResolvedValue([{ address, family: address.includes(':') ? 6 : 4 }]);
			await expect(validateDatabricksHost('https://internal.example.com')).rejects.toThrow(
				'must be a public Databricks workspace URL',
			);
		});

		it('should reject when any resolved address is private', async () => {
			mockedLookup.mockResolvedValue([
				{ address: '52.1.2.3', family: 4 },
				{ address: '10.0.0.1', family: 4 },
			]);
			await expect(validateDatabricksHost('https://rebind.example.com')).rejects.toThrow(
				'must be a public Databricks workspace URL',
			);
		});

		it('should reject an IPv4-mapped IPv6 address pointing at a private target', async () => {
			mockedLookup.mockResolvedValue([{ address: '::ffff:169.254.169.254', family: 6 }]);
			await expect(validateDatabricksHost('https://mapped.example.com')).rejects.toThrow(
				'must be a public Databricks workspace URL',
			);
		});

		it('should reject when the host cannot be resolved', async () => {
			mockedLookup.mockRejectedValue(new Error('ENOTFOUND'));
			await expect(validateDatabricksHost('https://does-not-exist.example.com')).rejects.toThrow(
				'Could not resolve',
			);
		});
	});

	describe('validateResourceName', () => {
		it('should accept a valid endpoint name', () => {
			expect(validateResourceName('databricks-claude-sonnet-4-6', 'Model endpoint')).toBe(
				'databricks-claude-sonnet-4-6',
			);
		});

		it('should reject an empty name', () => {
			expect(() => validateResourceName('  ', 'Model endpoint')).toThrow('cannot be empty');
		});

		it('should reject a name with invalid characters', () => {
			expect(() => validateResourceName('../etc/passwd', 'Model endpoint')).toThrow(
				'invalid characters',
			);
		});
	});
});
