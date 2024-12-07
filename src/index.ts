import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';
import { ToolbarButton } from '@jupyterlab/apputils';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { AnalysisServer } from './server';

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/auto-analyze:plugin',
  autoStart: true,
  requires: [INotebookTracker, ISettingRegistry],
  activate: async (
    app: JupyterFrontEnd,
    notebooks: INotebookTracker,
    settings: ISettingRegistry
  ) => {
    console.log('JupyterLab extension auto-analyze is activated!');

    const analyzeServer = new AnalysisServer(settings);

    notebooks.widgetAdded.connect((_, panel) => {
      const button = new ToolbarButton({
        className: 'jp-AutoAnalyze-button',
        label: 'Auto Analyze',
        onClick: () => analyzeServer.analyze(panel)
      });
      panel.toolbar.insertItem(10, 'autoAnalyze', button);
    });
  }
};

export default plugin;