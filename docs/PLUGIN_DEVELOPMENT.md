# Plugin development

TEN supports external AI providers, market-data providers, analysis engines, and future notification providers through the `Plugin` lifecycle interfaces.

## Package contract

Implement one of:

- `AIProviderPlugin`
- `MarketDataProviderPlugin`
- `AnalysisEnginePlugin`
- `NotificationProviderPlugin`

Expose a no-argument factory through the package entry point group:

```toml
[project.entry-points."ten.plugins"]
my_provider = "my_ten_plugin:create_plugin"
```

The factory must return a `Plugin` with validated `PluginMetadata`. `PluginLoader` discovers installed entry points; `PluginRegistry` rejects duplicate `(type, name)` identities. Secrets remain environment-backed and must never appear in plugin metadata or YAML.

AI plugins receive a `FeatureSnapshot`. Market-data plugins return provider records to a normalization adapter. Notification plugins consume named events but cannot alter pipeline results.
