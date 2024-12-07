declare module "@modelcontextprotocol/sdk" {
    class Server {
        constructor(config: {
            name: string;
            version: string;
            capabilities?: {
                tools?: {};
            };
        });
        onerror: (error: Error) => void;
        setRequestHandler: (schema: any, handler: Function) => void;
        close: () => Promise<void>;
    }
    const ListToolsRequestSchema: string;
    const CallToolRequestSchema: string;
    enum ErrorCode {
        InvalidRequest = "INVALID_REQUEST",
        MethodNotFound = "METHOD_NOT_FOUND",
        InvalidParams = "INVALID_PARAMS",
        InternalError = "INTERNAL_ERROR"
    }
    class McpError extends Error {
        constructor(code: ErrorCode, message: string);
    }
}
