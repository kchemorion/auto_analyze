"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AnalysisServer = void 0;
const sdk_1 = require("@modelcontextprotocol/sdk");
const sdk_2 = require("@modelcontextprotocol/sdk");
class AnalysisServer {
    // In the constructor
    constructor(settingRegistry) {
        this.currentNotebook = null;
        this.dataSourceConfig = null;
        this.lastCellOutput = null;
        this.variableCache = new Map();
        this.settingRegistry = settingRegistry;
        this.server = new sdk_1.Server({
            name: "jupyter-analyst",
            version: "0.1.0",
            capabilities: {
                tools: {}
            }
        });
        // Use the settingRegistry
        this.settingRegistry.load("@jupyterlab/auto-analyze:plugin")
            .then(settings => {
            // Use settings as needed
            this.updateSettings(settings);
        })
            .catch(error => console.error("Failed to load settings:", error));
        const defaultConfig = {
            type: 'file',
            connection: {}
        };
        this.dataSourceConfig = defaultConfig;
        this.setupTools();
        this.setupErrorHandling();
    }
    updateSettings(settings) {
        // Update and use dataSourceConfig
        if (settings.get('defaultDataSource')) {
            this.dataSourceConfig = settings.get('defaultDataSource');
            // Use it to configure something
            if (this.dataSourceConfig) {
                this.handleDataSourceConnection(this.dataSourceConfig).catch(error => console.error("Failed to connect to default data source:", error));
            }
        }
        this.variableCache.set('settings', settings);
    }
    setupErrorHandling() {
        this.server.onerror = (error) => {
            console.error("[MCP Error]", error);
        };
        process.on('SIGINT', async () => {
            if (this.currentNotebook) {
                await this.currentNotebook.context.save();
            }
            process.exit(0);
        });
    }
    setupTools() {
        this.server.setRequestHandler(sdk_2.ListToolsRequestSchema, async () => ({
            tools: [
                // Data Source Connection Tools
                {
                    name: "connect_datasource",
                    description: "Connect to a data source (file, database, API, or cloud service)",
                    inputSchema: {
                        type: "object",
                        properties: {
                            type: {
                                type: "string",
                                enum: ["file", "database", "api", "bigquery", "snowflake", "postgres", "mysql"],
                                description: "Type of data source to connect to"
                            },
                            connection: {
                                type: "object",
                                description: "Connection parameters specific to the data source type"
                            }
                        },
                        required: ["type", "connection"]
                    }
                },
                // Cell Execution Tools
                {
                    name: "execute_cell",
                    description: "Execute a notebook cell and return its output",
                    inputSchema: {
                        type: "object",
                        properties: {
                            code: {
                                type: "string",
                                description: "Code or text content for the cell"
                            },
                            cell_type: {
                                type: "string",
                                enum: ["code", "markdown"],
                                description: "Type of cell to execute"
                            }
                        },
                        required: ["code", "cell_type"]
                    }
                },
                // Data Inspection Tools
                {
                    name: "inspect_data",
                    description: "Get detailed information about a dataset",
                    inputSchema: {
                        type: "object",
                        properties: {
                            variable_name: {
                                type: "string",
                                description: "Name of the variable to inspect"
                            },
                            analysis_type: {
                                type: "string",
                                enum: ["basic", "detailed", "statistical", "missing"],
                                description: "Type of analysis to perform"
                            }
                        },
                        required: ["variable_name"]
                    }
                },
                {
                    name: "generate_profile_report",
                    description: "Generate a comprehensive data profile report",
                    inputSchema: {
                        type: "object",
                        properties: {
                            variable_name: {
                                type: "string",
                                description: "Variable containing the dataset to profile"
                            },
                            minimal: {
                                type: "boolean",
                                description: "Generate minimal report for large datasets"
                            }
                        },
                        required: ["variable_name"]
                    }
                },
                // Visualization Tools
                {
                    name: "create_visualization",
                    description: "Create various types of visualizations using plotly",
                    inputSchema: {
                        type: "object",
                        properties: {
                            plot_type: {
                                type: "string",
                                enum: [
                                    "scatter", "line", "bar", "histogram", "box",
                                    "violin", "heatmap", "correlation", "pair_plot",
                                    "3d_scatter", "contour", "distribution"
                                ]
                            },
                            data_vars: {
                                type: "object",
                                description: "Variables and columns to plot"
                            },
                            layout: {
                                type: "object",
                                description: "Plot layout configuration"
                            }
                        },
                        required: ["plot_type", "data_vars"]
                    }
                },
                // Statistical Analysis Tools
                {
                    name: "run_statistical_test",
                    description: "Perform statistical tests on the data",
                    inputSchema: {
                        type: "object",
                        properties: {
                            test_type: {
                                type: "string",
                                enum: [
                                    "normality", "ttest", "anova", "chi_square",
                                    "correlation", "regression", "mann_whitney",
                                    "kruskal_wallis", "feature_importance"
                                ]
                            },
                            variables: {
                                type: "array",
                                items: { type: "string" }
                            },
                            params: {
                                type: "object",
                                description: "Additional test parameters"
                            }
                        },
                        required: ["test_type", "variables"]
                    }
                },
                // Feature Engineering Tools
                {
                    name: "engineer_features",
                    description: "Perform feature engineering operations",
                    inputSchema: {
                        type: "object",
                        properties: {
                            operation: {
                                type: "string",
                                enum: [
                                    "encoding", "scaling", "binning", "interactions",
                                    "dimension_reduction", "feature_selection",
                                    "text_vectorization", "date_features"
                                ]
                            },
                            columns: {
                                type: "array",
                                items: { type: "string" }
                            },
                            params: {
                                type: "object",
                                description: "Operation-specific parameters"
                            }
                        },
                        required: ["operation", "columns"]
                    }
                },
                // Data Cleaning Tools
                {
                    name: "clean_data",
                    description: "Perform data cleaning operations",
                    inputSchema: {
                        type: "object",
                        properties: {
                            operation: {
                                type: "string",
                                enum: [
                                    "handle_missing", "remove_duplicates",
                                    "handle_outliers", "fix_datatypes",
                                    "normalize_text", "validate_values"
                                ]
                            },
                            columns: {
                                type: "array",
                                items: { type: "string" }
                            },
                            params: {
                                type: "object",
                                description: "Cleaning parameters"
                            }
                        },
                        required: ["operation", "columns"]
                    }
                },
                // Export and Reporting Tools
                {
                    name: "export_results",
                    description: "Export analysis results in various formats",
                    inputSchema: {
                        type: "object",
                        properties: {
                            format: {
                                type: "string",
                                enum: ["html", "pdf", "json", "csv", "excel"]
                            },
                            content: {
                                type: "object",
                                description: "Content to export"
                            },
                            filename: {
                                type: "string"
                            }
                        },
                        required: ["format", "content"]
                    }
                }
            ]
        }));
        // Tool Handlers Implementation
        this.server.setRequestHandler(sdk_2.CallToolRequestSchema, async (request) => {
            if (!this.currentNotebook) {
                return {
                    content: [{
                            type: "text",
                            text: "No active notebook"
                        }],
                    isError: true
                };
            }
            try {
                switch (request.params.name) {
                    case "connect_datasource":
                        return await this.handleDataSourceConnection(request.params.arguments);
                    case "execute_cell":
                        return await this.executeCell(request.params.arguments);
                    case "inspect_data":
                        return await this.inspectData(request.params.arguments);
                    case "generate_profile_report":
                        return await this.generateProfileReport(request.params.arguments);
                    case "create_visualization":
                        return await this.createVisualization(request.params.arguments);
                    case "run_statistical_test":
                        return await this.runStatisticalTest(request.params.arguments);
                    case "engineer_features":
                        return await this.engineerFeatures(request.params.arguments);
                    case "clean_data":
                        return await this.cleanData(request.params.arguments);
                    case "export_results":
                        return await this.exportResults(request.params.arguments);
                    default:
                        throw new sdk_2.McpError(sdk_2.ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
                }
            }
            catch (error) {
                const err = error;
                return {
                    content: [{
                            type: "text",
                            text: `Error: ${err.message}`
                        }],
                    isError: true
                };
            }
        });
    }
    // Data Source Connection Handler
    async handleDataSourceConnection(params) {
        const setupCode = this.generateConnectionCode(params);
        try {
            const result = await this.executeCell({
                cell_type: 'code',
                code: setupCode
            });
            // Store configuration for later use
            this.dataSourceConfig = params;
            this.variableCache.set('currentDataSource', params);
            if (result && result.content) {
                const output = this.lastCellOutput;
                this.variableCache.set('connectionOutput', output);
            }
            return {
                content: [{
                        type: "text",
                        text: "Data source connected successfully"
                    }]
            };
        }
        catch (error) {
            const err = error;
            throw new sdk_2.McpError(sdk_2.ErrorCode.InternalError, `Failed to connect to data source: ${err.message}`);
        }
    }
    // Remove unused handleError method if not needed
    generateConnectionCode(config) {
        switch (config.type) {
            case 'file':
                return `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_${config.connection.format || 'csv'}('${config.connection.path}')
`;
            case 'postgres':
                return `
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://${config.connection.username}:${config.connection.password}@${config.connection.host}:${config.connection.port}/${config.connection.database}')
df = pd.read_sql('${config.connection.query}', engine)
`;
            case 'bigquery':
                return `
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project='${config.connection.projectId}')
query = '''${config.connection.query}'''
df = client.query(query).to_dataframe()
`;
            case 'snowflake':
                return `
import snowflake.connector
import pandas as pd

ctx = snowflake.connector.connect(
  user='${config.connection.username}',
  password='${config.connection.password}',
  account='${config.connection.host}',
  warehouse='${config.connection.warehouse}',
  database='${config.connection.database}',
  schema='${config.connection.schema}'
)
df = pd.read_sql('${config.connection.query}', ctx)
`;
            default:
                throw new Error(`Unsupported data source type: ${config.type}`);
        }
    }
    // Data Inspection Handler
    async inspectData(params) {
        const inspectionCode = `
import pandas as pd
import numpy as np
from IPython.display import display, HTML

def analyze_dataset(df, analysis_type='basic'):
  info = {}
  
  if analysis_type in ['basic', 'all']:
      info['basic'] = {
          'shape': df.shape,
          'columns': list(df.columns),
          'dtypes': df.dtypes.to_dict(),
          'missing_values': df.isnull().sum().to_dict(),
          'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
      }
  
  if analysis_type in ['statistical', 'all']:
      numeric_cols = df.select_dtypes(include=[np.number]).columns
      info['statistical'] = {
          'description': df.describe().to_dict(),
          'skewness': df[numeric_cols].skew().to_dict(),
          'kurtosis': df[numeric_cols].kurtosis().to_dict(),
          'correlation': df[numeric_cols].corr().to_dict()
      }
  
  if analysis_type in ['categorical', 'all']:
      categorical_cols = df.select_dtypes(include=['object', 'category']).columns
      info['categorical'] = {
          'unique_counts': {col: df[col].nunique() for col in categorical_cols},
          'value_counts': {col: df[col].value_counts().to_dict() for col in categorical_cols}
      }
      
  return info

result = analyze_dataset(${params.variable_name}, '${params.analysis_type || 'basic'}')
print(result)
`;
        return await this.executeCell({
            cell_type: 'code',
            code: inspectionCode
        });
    }
    // Profile Report Generator
    async generateProfileReport(params) {
        const profileCode = `
from ydata_profiling import ProfileReport
import json

profile = ProfileReport(${params.variable_name}, minimal=${params.minimal || false})
profile.to_file('temp_profile.html')

with open('temp_profile.html', 'r') as f:
  html_content = f.read()
  
print(json.dumps({'profile_html': html_content}))
`;
        return await this.executeCell({
            cell_type: 'code',
            code: profileCode
        });
    } // Visualization Creator
    async createVisualization(params) {
        const plotCode = `
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_plot(data_vars, plot_type, layout=None):
  fig = None
  
  try:
      if plot_type == 'scatter':
          fig = px.scatter(data_frame=df, 
                         x=data_vars['x'], 
                         y=data_vars['y'],
                         color=data_vars.get('color'),
                         size=data_vars.get('size'),
                         hover_data=data_vars.get('hover_data'))

      elif plot_type == 'correlation':
          corr_matrix = df[data_vars['columns']].corr()
          fig = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r')

      elif plot_type == 'pair_plot':
          fig = px.scatter_matrix(df[data_vars['columns']],
                                dimensions=data_vars['columns'],
                                color=data_vars.get('color'))

      elif plot_type == 'distribution':
          fig = make_subplots(rows=len(data_vars['columns']), cols=1)
          for i, col in enumerate(data_vars['columns'], 1):
              fig.add_trace(
                  go.Histogram(x=df[col], name=col),
                  row=i, col=1
              )

      elif plot_type == '3d_scatter':
          fig = px.scatter_3d(df,
                            x=data_vars['x'],
                            y=data_vars['y'],
                            z=data_vars['z'],
                            color=data_vars.get('color'))

      elif plot_type == 'time_series':
          fig = px.line(df,
                       x=data_vars['x'],
                       y=data_vars['y'],
                       color=data_vars.get('color'))

      if layout:
          fig.update_layout(**layout)
          
      fig.write_html('temp_plot.html')
      with open('temp_plot.html', 'r') as f:
          return {'plot_html': f.read()}
          
  except Exception as e:
      return {'error': str(e)}

result = create_plot(${JSON.stringify(params.data_vars)}, 
                  '${params.plot_type}',
                  ${JSON.stringify(params.layout || {})})
print(result)
`;
        return await this.executeCell({
            cell_type: 'code',
            code: plotCode
        });
    }
    // Statistical Test Runner
    async runStatisticalTest(params) {
        const testCode = `
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def run_test(test_type, variables, params=None):
  results = {}
  
  try:
      if test_type == 'normality':
          for var in variables:
              stat, p_value = stats.normaltest(df[var].dropna())
              results[var] = {
                  'statistic': stat,
                  'p_value': p_value,
                  'is_normal': p_value > 0.05
              }

      elif test_type == 'ttest':
          stat, p_value = stats.ttest_ind(
              df[variables[0]].dropna(),
              df[variables[1]].dropna()
          )
          results = {
              'statistic': stat,
              'p_value': p_value,
              'significant': p_value < 0.05
          }

      elif test_type == 'correlation':
          corr_matrix = df[variables].corr()
          p_matrix = pd.DataFrame(
              [[stats.pearsonr(df[c1], df[c2])[1] 
                for c2 in variables] 
               for c1 in variables],
              columns=variables,
              index=variables
          )
          results = {
              'correlation_matrix': corr_matrix.to_dict(),
              'p_values': p_matrix.to_dict()
          }

      elif test_type == 'chi_square':
          contingency = pd.crosstab(df[variables[0]], df[variables[1]])
          chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
          results = {
              'chi2': chi2,
              'p_value': p_value,
              'dof': dof,
              'contingency_table': contingency.to_dict(),
              'significant': p_value < 0.05
          }

      elif test_type == 'feature_importance':
          target = variables[-1]
          features = variables[:-1]
          if df[target].dtype in ['int64', 'bool']:
              importance = mutual_info_classif(
                  df[features], df[target], random_state=42
              )
          else:
              importance = mutual_info_regression(
                  df[features], df[target], random_state=42
              )
          results = {
              'feature_importance': dict(zip(features, importance.tolist()))
          }

      return results
      
  except Exception as e:
      return {'error': str(e)}

results = run_test('${params.test_type}', 
                ${JSON.stringify(params.variables)},
                ${JSON.stringify(params.params || {})})
print(results)
`;
        return await this.executeCell({
            cell_type: 'code',
            code: testCode
        });
    }
    // Feature Engineering Handler
    async engineerFeatures(params) {
        const engineeringCode = `
from sklearn.preprocessing import (
  LabelEncoder, OneHotEncoder, StandardScaler, 
  MinMaxScaler, RobustScaler
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def engineer_features(df, operation, columns, params=None):
  try:
      if operation == 'encoding':
          if params.get('method') == 'label':
              le = LabelEncoder()
              for col in columns:
                  df[f'{col}_encoded'] = le.fit_transform(df[col])
          elif params.get('method') == 'onehot':
              enc = OneHotEncoder(sparse=False)
              encoded = enc.fit_transform(df[columns])
              encoded_df = pd.DataFrame(
                  encoded,
                  columns=enc.get_feature_names_out(columns)
              )
              df = pd.concat([df, encoded_df], axis=1)

      elif operation == 'scaling':
          if params.get('method') == 'standard':
              scaler = StandardScaler()
          elif params.get('method') == 'minmax':
              scaler = MinMaxScaler()
          elif params.get('method') == 'robust':
              scaler = RobustScaler()
              
          df[columns] = scaler.fit_transform(df[columns])

      elif operation == 'dimension_reduction':
          n_components = params.get('n_components', 2)
          pca = PCA(n_components=n_components)
          transformed = pca.fit_transform(df[columns])
          for i in range(n_components):
              df[f'PC{i+1}'] = transformed[:, i]
          explained_variance = pca.explained_variance_ratio_

      return {
          'df': df.to_dict('records'),
          'info': {
              'operation': operation,
              'columns_affected': columns,
              'new_columns': [col for col in df.columns if col not in columns]
          }
      }

  except Exception as e:
      return {'error': str(e)}

result = engineer_features(
  df,
  '${params.operation}',
  ${JSON.stringify(params.columns)},
  ${JSON.stringify(params.params || {})}
)
print(result)
`;
        return await this.executeCell({
            cell_type: 'code',
            code: engineeringCode
        });
    } // Data Cleaning Handler
    async cleanData(params) {
        const cleaningCode = `
import pandas as pd
import numpy as np
from scipy import stats

def clean_data(df, operation, columns, params=None):
  try:
      df_cleaned = df.copy()
      
      if operation == 'handle_missing':
          strategy = params.get('strategy', 'mean')
          for col in columns:
              if strategy == 'mean':
                  df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
              elif strategy == 'median':
                  df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
              elif strategy == 'mode':
                  df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
              elif strategy == 'forward':
                  df_cleaned[col].fillna(method='ffill', inplace=True)
              elif strategy == 'backward':
                  df_cleaned[col].fillna(method='bfill', inplace=True)

      elif operation == 'handle_outliers':
          method = params.get('method', 'zscore')
          threshold = params.get('threshold', 3)
          
          for col in columns:
              if method == 'zscore':
                  z_scores = np.abs(stats.zscore(df_cleaned[col]))
                  df_cleaned[col] = np.where(z_scores > threshold, 
                                           df_cleaned[col].mean(), 
                                           df_cleaned[col])
              elif method == 'iqr':
                  Q1 = df_cleaned[col].quantile(0.25)
                  Q3 = df_cleaned[col].quantile(0.75)
                  IQR = Q3 - Q1
                  df_cleaned[col] = np.where(
                      (df_cleaned[col] < (Q1 - 1.5 * IQR)) | 
                      (df_cleaned[col] > (Q3 + 1.5 * IQR)),
                      df_cleaned[col].median(),
                      df_cleaned[col]
                  )

      elif operation == 'fix_datatypes':
          for col in columns:
              target_type = params.get('types', {}).get(col)
              if target_type:
                  if target_type == 'datetime':
                      df_cleaned[col] = pd.to_datetime(df_cleaned[col])
                  elif target_type == 'numeric':
                      df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                  elif target_type == 'category':
                      df_cleaned[col] = df_cleaned[col].astype('category')

      elif operation == 'normalize_text':
          for col in columns:
              df_cleaned[col] = df_cleaned[col].str.lower()
              df_cleaned[col] = df_cleaned[col].str.strip()
              if params.get('remove_punctuation'):
                  df_cleaned[col] = df_cleaned[col].str.replace('[^\w\s]', '')

      return {
          'cleaned_data': df_cleaned.to_dict('records'),
          'info': {
              'rows_affected': len(df) - len(df_cleaned),
              'null_counts': df_cleaned[columns].isnull().sum().to_dict()
          }
      }

  except Exception as e:
      return {'error': str(e)}

result = clean_data(
  df,
  '${params.operation}',
  ${JSON.stringify(params.columns)},
  ${JSON.stringify(params.params || {})}
)
print(result)
`;
        return await this.executeCell({
            cell_type: 'code',
            code: cleaningCode
        });
    }
    // Export Handler
    async exportResults(params) {
        const exportCode = `
import pandas as pd
import json
import plotly.io as pio
from fpdf import FPDF
import base64

def export_content(content, format, filename):
  try:
      if format == 'html':
          with open(f'{filename}.html', 'w') as f:
              f.write(content)
      
      elif format == 'pdf':
          pdf = FPDF()
          pdf.add_page()
          pdf.set_font('Arial', size=12)
          pdf.multi_cell(0, 10, txt=content)
          pdf.output(f'{filename}.pdf')
      
      elif format == 'json':
          with open(f'{filename}.json', 'w') as f:
              json.dump(content, f, indent=2)
      
      elif format == 'csv':
          df = pd.DataFrame(content)
          df.to_csv(f'{filename}.csv', index=False)
      
      elif format == 'excel':
          df = pd.DataFrame(content)
          df.to_excel(f'{filename}.xlsx', index=False)
      
      return {'message': f'Successfully exported to {filename}.{format}'}
      
  except Exception as e:
      return {'error': str(e)}

result = export_content(
  ${JSON.stringify(params.content)},
  '${params.format}',
  '${params.filename}'
)
print(result)
`;
        return await this.executeCell({
            cell_type: 'code',
            code: exportCode
        });
    }
    // Helper Utilities
    async ensureNotebookReady() {
        if (!this.currentNotebook) {
            throw new Error("No active notebook");
        }
        await this.currentNotebook.sessionContext.ready;
        await this.currentNotebook.context.ready;
    }
    async executeCell(params) {
        var _a, _b, _c;
        if (!((_a = this.currentNotebook) === null || _a === void 0 ? void 0 : _a.model)) {
            throw new Error("No notebook model");
        }
        // const model = this.currentNotebook.model;
        // Create new cell
        // const cell = model.sharedModel.insertCell(
        //   model.sharedModel.cells.length,
        //   {
        //     cell_type: params.cell_type,
        //     source: params.code
        //   }
        // );
        if (params.cell_type === 'code') {
            try {
                const future = (_c = (_b = this.currentNotebook.sessionContext.session) === null || _b === void 0 ? void 0 : _b.kernel) === null || _c === void 0 ? void 0 : _c.requestExecute({
                    code: params.code
                });
                if (future) {
                    const output = await future.done;
                    this.lastCellOutput = output;
                    this.variableCache.set('lastOutput', output);
                    return {
                        content: [{
                                type: "text",
                                text: JSON.stringify(output)
                            }]
                    };
                }
            }
            catch (error) {
                const err = error;
                return {
                    content: [{
                            type: "text",
                            text: `Execution error: ${err.message}`
                        }],
                    isError: true
                };
            }
        }
        return {
            content: [{
                    type: "text",
                    text: "Cell created"
                }]
        };
    }
    // Error Handler Utilities
    // private handleError(error: any): any {
    //   console.error("[Analysis Error]", error);
    //   // Create markdown cell with error details
    //   this.executeCell({
    //     cell_type: 'markdown',
    //     code: `## ⚠️ Error\n\n${error.message}\n\n### Details\n\`\`\`\n${error.stack}\n\`\`\``
    //   });
    //   return {
    //     content: [{
    //       type: "text",
    //       text: `Error: ${error.message}`
    //     }],
    //     isError: true
    //   };
    // }
    // Main Analysis Entry Point
    async analyze(notebook) {
        this.currentNotebook = notebook;
        await this.ensureNotebookReady();
        // Initialize analysis environment
        await this.executeCell({
            cell_type: 'markdown',
            code: '# 📊 Automated Data Analysis Report\n\n_Generated by JupyterLab Auto Analyze_'
        });
    }
}
exports.AnalysisServer = AnalysisServer;
exports.default = AnalysisServer;
