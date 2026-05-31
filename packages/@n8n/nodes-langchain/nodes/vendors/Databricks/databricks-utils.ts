/**
 * Shared Databricks validation utilities for all Databricks LangChain nodes.
 */

import ipaddr from 'ipaddr.js';
import { UserError, type ILoadOptionsFunctions, type INodePropertyOptions } from 'n8n-workflow';
import { lookup } from 'node:dns/promises';

/**
 * Address ranges that must never be the target of an outbound request.
 * `ipaddr.js` classifies any normal, publicly-routable address as `unicast`;
 * everything else (loopback, private, link-local, carrier-grade NAT,
 * IPv4-mapped, reserved, multicast, …) is rejected.
 */
function assertPublicAddress(address: string): void {
	let parsed: ipaddr.IPv4 | ipaddr.IPv6;
	try {
		parsed = ipaddr.parse(address);
	} catch {
		// If we cannot parse what the resolver returned, fail closed.
		throw new UserError(
			'Databricks host must be a public Databricks workspace URL, not a local or private address.',
		);
	}

	// Resolve IPv4-mapped IPv6 addresses (e.g. ::ffff:169.254.169.254) to their
	// underlying IPv4 address so the classification reflects the real target.
	if (parsed.kind() === 'ipv6' && (parsed as ipaddr.IPv6).isIPv4MappedAddress()) {
		parsed = (parsed as ipaddr.IPv6).toIPv4Address();
	}

	if (parsed.range() !== 'unicast') {
		throw new UserError(
			'Databricks host must be a public Databricks workspace URL, not a local or private address.',
		);
	}
}

/**
 * Validates and sanitizes a Databricks host URL.
 * Ensures HTTPS protocol, strips trailing slashes, and resolves the hostname to
 * verify it does not point at a private, loopback, or metadata address. Resolving
 * the hostname (rather than string-matching it) also covers alternate IP
 * encodings and hostnames that resolve to internal addresses.
 * @throws UserError if the host is not a valid, public HTTPS URL
 */
export async function validateDatabricksHost(host: string): Promise<string> {
	const sanitized = host.replace(/\/+$/, '');

	let url: URL;
	try {
		url = new URL(sanitized);
	} catch {
		throw new UserError(
			'Databricks host must be a valid URL, e.g. https://your-workspace.cloud.databricks.com.',
		);
	}

	if (url.protocol !== 'https:') {
		throw new UserError(
			'Databricks host must use HTTPS. Please update your credentials to use https://.',
		);
	}

	// `URL.hostname` wraps IPv6 literals in brackets, which the resolver rejects.
	const hostname = url.hostname.replace(/^\[|\]$/g, '');

	let addresses: Array<{ address: string }>;
	try {
		addresses = await lookup(hostname, { all: true, verbatim: true });
	} catch {
		throw new UserError(`Could not resolve Databricks host "${hostname}".`);
	}

	if (addresses.length === 0) {
		throw new UserError(`Could not resolve Databricks host "${hostname}".`);
	}

	for (const { address } of addresses) {
		assertPublicAddress(address);
	}

	return sanitized;
}

/**
 * Validates that a Databricks resource name (such as a serving endpoint name) is
 * non-empty and made up of safe characters. Acts as an early input guard before
 * the value is sent to Databricks.
 * Allowed characters: alphanumeric, hyphens, underscores, dots.
 * @throws UserError if the name is empty or contains unexpected characters
 */
export function validateResourceName(name: string, resourceType: string): string {
	if (!name || name.trim().length === 0) {
		throw new UserError(`${resourceType} name cannot be empty.`);
	}

	// Allow three-level namespace format for index names: catalog.schema.index_name
	// and simple names for endpoints: my-endpoint-name
	const safePattern = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;

	if (!safePattern.test(name)) {
		throw new UserError(
			`${resourceType} name "${name}" contains invalid characters. Only alphanumeric, hyphens, underscores, and dots are allowed.`,
		);
	}

	if (name.includes('..') || name.includes('//')) {
		throw new UserError(`${resourceType} name "${name}" contains invalid characters.`);
	}

	return name;
}

/**
 * Loads the Databricks Model Serving endpoints for a `loadOptions` model dropdown.
 * Shared by the Databricks chat-model and embeddings nodes so the host is validated
 * and the endpoint list is built the same way for both.
 */
export async function loadDatabricksServingEndpoints(
	this: ILoadOptionsFunctions,
): Promise<INodePropertyOptions[]> {
	const credentials = await this.getCredentials<{ host: string }>('databricksApi');
	const host = await validateDatabricksHost(credentials.host);

	const response = (await this.helpers.httpRequestWithAuthentication.call(this, 'databricksApi', {
		method: 'GET',
		baseURL: host,
		url: '/api/2.0/serving-endpoints',
	})) as { endpoints?: Array<{ name: string }> };

	return (response.endpoints ?? [])
		.map((endpoint) => ({ name: endpoint.name, value: endpoint.name }))
		.sort((a, b) => a.name.localeCompare(b.name));
}
