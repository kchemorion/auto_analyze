"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const notebook_1 = require("@jupyterlab/notebook");
const apputils_1 = require("@jupyterlab/apputils");
const settingregistry_1 = require("@jupyterlab/settingregistry");
const server_1 = require("./server");
const plugin = {
    id: '@jupyterlab/auto-analyze:plugin',
    autoStart: true,
    requires: [notebook_1.INotebookTracker, settingregistry_1.ISettingRegistry],
    activate: async (app, notebooks, settings) => {
        console.log('JupyterLab extension auto-analyze is activated!');
        const analyzeServer = new server_1.AnalysisServer(settings);
        notebooks.widgetAdded.connect((_, panel) => {
            const button = new apputils_1.ToolbarButton({
                className: 'jp-AutoAnalyze-button',
                label: 'Auto Analyze',
                onClick: () => analyzeServer.analyze(panel)
            });
            panel.toolbar.insertItem(10, 'autoAnalyze', button);
        });
    }
};
exports.default = plugin;
