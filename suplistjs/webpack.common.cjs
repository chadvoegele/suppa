const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const { CleanWebpackPlugin } = require('clean-webpack-plugin')

module.exports = {
  entry: {
    index: './src/index.js'
  },
  resolve: {
    alias: {
      '@suplistmlrs': path.resolve(__dirname, '../suplistmlrs/build/wasm.js')
    }
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader']
      }
    ]
  },
  plugins: [
    new CleanWebpackPlugin(),
    new HtmlWebpackPlugin({
      template: './src/templates/template.html'
    })
  ],
  output: {
    filename: '[name].[fullhash].bundle.js',
    path: path.resolve(__dirname, 'dist')
  },
  performance: {
    maxAssetSize: 16777216
  }
}
