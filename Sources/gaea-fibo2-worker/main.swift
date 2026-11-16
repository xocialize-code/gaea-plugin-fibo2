import Fibo2Plugin
import GaeaPluginKit

// The plugin's worker: Fibo2Plugin's definition over GPP (GaeaPluginKit). `--manifest <path>` writes manifest.json.
await GaeaPluginMain.run(fibo2Plugin)
