const { getDefaultConfig } = require("expo/metro-config");
const path = require("path");

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Alias @/ → ./src/ (Metro SDK 52 natif, sans babel-plugin-module-resolver)
config.resolver.alias = {
  "@": path.resolve(__dirname, "src"),
};

module.exports = config;
