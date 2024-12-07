declare module "@modelcontextprotocol/sdk" {
    export class Server {
        constructor(config: { name: string; version: string; capabilities?: { tools?: {} } });
        onerror: (error: Error) => void;
        setRequestHandler: (schema: any, handler: Function) => void;
        close: () => Promise<void>;
    }

    export const ListToolsRequestSchema: string;
    export const CallToolRequestSchema: string;
    
    export enum ErrorCode {
        InvalidRequest = 'INVALID_REQUEST',
        MethodNotFound = 'METHOD_NOT_FOUND',
        InvalidParams = 'INVALID_PARAMS',
        InternalError = 'INTERNAL_ERROR'
    }

    export class McpError extends Error {
        constructor(code: ErrorCode, message: string);
    }
}