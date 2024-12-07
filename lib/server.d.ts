import { NotebookPanel } from '@jupyterlab/notebook';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
export declare class AnalysisServer {
    private server;
    private currentNotebook;
    private settingRegistry;
    private dataSourceConfig;
    private lastCellOutput;
    private variableCache;
    constructor(settingRegistry: ISettingRegistry);
    private updateSettings;
    private setupErrorHandling;
    private setupTools;
    private handleDataSourceConnection;
    private generateConnectionCode;
    private inspectData;
    private generateProfileReport;
    private createVisualization;
    private runStatisticalTest;
    private engineerFeatures;
    private cleanData;
    private exportResults;
    private ensureNotebookReady;
    private executeCell;
    analyze(notebook: NotebookPanel): Promise<void>;
}
export default AnalysisServer;
