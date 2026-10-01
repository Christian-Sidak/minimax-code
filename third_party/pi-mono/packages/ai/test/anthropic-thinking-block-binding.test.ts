import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { describe, expect, it } from "vitest";
import {
	streamAnthropic,
	THINKING_BLOCK_BINDING_MODEL_IDS,
	supportsThinkingBlockBinding,
} from "../src/providers/anthropic.ts";
import type { AnthropicOptions } from "../src/providers/anthropic.ts";
import type { Context, Model } from "../src/types.ts";

interface CapturedRequest {
	headers: IncomingMessage["headers"];
	body: Record<string, unknown>;
}

function createModel(
	baseUrl: string,
	modelId: string,
	compat?: Model<"anthropic-messages">["compat"],
): Model<"anthropic-messages"> {
	return {
		id: modelId,
		name: modelId,
		api: "anthropic-messages",
		provider: "test-anthropic",
		baseUrl,
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 32000,
		compat,
	};
}

function makeContext(): Context {
	return {
		messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
	};
}

async function readRequestBody(request: IncomingMessage): Promise<Record<string, unknown>> {
	const chunks: Buffer[] = [];
	for await (const chunk of request) {
		chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
	}
	return JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
}

function writeEmptySseResponse(response: ServerResponse): void {
	response.writeHead(200, { "content-type": "text/event-stream" });
	response.end();
}

async function captureAnthropicRequest(
	model: Model<"anthropic-messages">,
	options?: Omit<AnthropicOptions, "apiKey">,
): Promise<CapturedRequest> {
	let capturedRequest: CapturedRequest | undefined;

	const server = createServer(async (request, response) => {
		capturedRequest = {
			headers: request.headers,
			body: await readRequestBody(request),
		};
		writeEmptySseResponse(response);
	});

	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address() as AddressInfo;
	const modelWithUrl = { ...model, baseUrl: `http://127.0.0.1:${address.port}` };

	try {
		const stream = streamAnthropic(modelWithUrl, makeContext(), {
			...options,
			apiKey: "test-key",
			cacheRetention: "none",
		});

		for await (const event of stream) {
			if (event.type === "done" || event.type === "error") break;
		}
	} finally {
		await new Promise<void>((resolve, reject) => {
			server.close((error) => (error ? reject(error) : resolve()));
		});
	}

	if (!capturedRequest) {
		throw new Error("Anthropic request was not captured");
	}
	return capturedRequest;
}

describe("supportsThinkingBlockBinding", () => {
	it("returns true for claude-opus-5-5", () => {
		const model = createModel("http://127.0.0.1:9", "claude-opus-5-5");
		expect(supportsThinkingBlockBinding(model)).toBe(true);
	});

	it("returns true for claude-sonnet-5-5", () => {
		const model = createModel("http://127.0.0.1:9", "claude-sonnet-5-5");
		expect(supportsThinkingBlockBinding(model)).toBe(true);
	});

	it("returns true for claude-fable-5-1", () => {
		const model = createModel("http://127.0.0.1:9", "claude-fable-5-1");
		expect(supportsThinkingBlockBinding(model)).toBe(true);
	});

	it("returns true for claude-mythos-5-1", () => {
		const model = createModel("http://127.0.0.1:9", "claude-mythos-5-1");
		expect(supportsThinkingBlockBinding(model)).toBe(true);
	});

	it("returns false for claude-opus-4-8 (older model without binding support)", () => {
		const model = createModel("http://127.0.0.1:9", "claude-opus-4-8");
		expect(supportsThinkingBlockBinding(model)).toBe(false);
	});

	it("returns false for claude-fable-5 (5.0, not 5.1)", () => {
		const model = createModel("http://127.0.0.1:9", "claude-fable-5");
		expect(supportsThinkingBlockBinding(model)).toBe(false);
	});

	it("respects compat.supportsThinkingBlockBinding override to true", () => {
		const model = createModel("http://127.0.0.1:9", "custom-proxy-model", {
			supportsThinkingBlockBinding: true,
		});
		expect(supportsThinkingBlockBinding(model)).toBe(true);
	});

	it("respects compat.supportsThinkingBlockBinding override to false on a binding model", () => {
		const model = createModel("http://127.0.0.1:9", "claude-opus-5-5", {
			supportsThinkingBlockBinding: false,
		});
		expect(supportsThinkingBlockBinding(model)).toBe(false);
	});

	it("covers all THINKING_BLOCK_BINDING_MODEL_IDS entries", () => {
		for (const id of THINKING_BLOCK_BINDING_MODEL_IDS) {
			const model = createModel("http://127.0.0.1:9", id);
			expect(supportsThinkingBlockBinding(model), `expected true for ${id}`).toBe(true);
		}
	});
});

describe("Anthropic thinking block binding request wiring", () => {
	it("sends thinking-binding-controls beta header for claude-opus-5-5", async () => {
		const model = createModel("http://placeholder", "claude-opus-5-5");
		const request = await captureAnthropicRequest(model, { thinkingEnabled: true });

		expect(request.headers["anthropic-beta"]).toContain("thinking-binding-controls-2026-08-01");
	});

	it("sends thinking-binding-controls beta header for claude-sonnet-5-5", async () => {
		const model = createModel("http://placeholder", "claude-sonnet-5-5");
		const request = await captureAnthropicRequest(model, { thinkingEnabled: true });

		expect(request.headers["anthropic-beta"]).toContain("thinking-binding-controls-2026-08-01");
	});

	it("does not send thinking-binding-controls beta header for claude-opus-4-8", async () => {
		const model = createModel("http://placeholder", "claude-opus-4-8", { forceAdaptiveThinking: true });
		const request = await captureAnthropicRequest(model, { thinkingEnabled: true });

		expect(request.headers["anthropic-beta"] ?? "").not.toContain("thinking-binding-controls-2026-08-01");
	});

	it("sends block_binding.prefix_mismatch_behavior=drop_block in thinking for claude-opus-5-5", async () => {
		const model = createModel("http://placeholder", "claude-opus-5-5");
		const request = await captureAnthropicRequest(model, { thinkingEnabled: true });

		const thinking = request.body.thinking as Record<string, unknown> | undefined;
		expect(thinking).toBeDefined();
		expect(thinking?.type).toBe("adaptive");
		expect((thinking?.block_binding as Record<string, unknown> | undefined)?.prefix_mismatch_behavior).toBe(
			"drop_block",
		);
	});

	it("does not include block_binding in thinking for non-binding models", async () => {
		const model = createModel("http://placeholder", "claude-opus-4-8", { forceAdaptiveThinking: true });
		const request = await captureAnthropicRequest(model, { thinkingEnabled: true });

		const thinking = request.body.thinking as Record<string, unknown> | undefined;
		expect(thinking?.type).toBe("adaptive");
		expect(thinking?.block_binding).toBeUndefined();
	});

	it("omits block_binding when compat.supportsThinkingBlockBinding is forced off", async () => {
		const model = createModel("http://placeholder", "claude-opus-5-5", {
			supportsThinkingBlockBinding: false,
		});
		const request = await captureAnthropicRequest(model, { thinkingEnabled: true });

		const thinking = request.body.thinking as Record<string, unknown> | undefined;
		expect(thinking?.block_binding).toBeUndefined();
		expect(request.headers["anthropic-beta"] ?? "").not.toContain("thinking-binding-controls-2026-08-01");
	});
});
