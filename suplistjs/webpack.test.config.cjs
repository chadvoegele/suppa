const path = require('path')
const { CleanWebpackPlugin } = require('clean-webpack-plugin')
const nodeExternals = require('webpack-node-externals')

module.exports = {
  mode: 'development',
  entry: {
    index: './test/test.js'
  },
  externalsPresets: { node: true },
  externals: [nodeExternals()],
  devtool: 'inline-source-map',
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['css-loader']
      }
    ]
  },
  plugins: [
    new CleanWebpackPlugin()
  ],
  output: {
    filename: 'test.[fullhash].bundle.cjs',
    path: path.resolve(__dirname, 'dist')
  }
}
