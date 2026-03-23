# TinyForge

A from-scratch **ONNX parser** in C++ (protobuf) plus a **browser demo** for **MobileNetV2**: explore the graph, read the story panels, and run inference with a small **WebAssembly** engine. There is no TensorFlow/PyTorch runtime in the browser—only your code and the weights.

The parser reads the **MobileNetV2** graph and exports `graph.json` and `weights.bin`. The web app visualizes the graph and runs a **WASM**-based inference demo. Goal: no magic—show what happens layer by layer.

---

## Repository layout

| Path | Description |
|------|-------------|
| `parser/` | ONNX → internal model, JSON and weight export |
| `engine/inference.cpp` | Built with Emscripten to `web/inference.js` + `inference.wasm` |
| `web/` | UI, graph, inference walkthrough, markdown stories |
| `assets/` | `graph.json`, `weights.bin`, labels, ONNX sources (for the parser) |

---

## Requirements (native build)

- **C++17**, `g++` or compatible compiler  
- **protobuf** (`libprotobuf`, via `pkg-config`)  
- To regenerate protobuf sources from `onnx.proto` → `onnx.pb.cc` / `onnx.pb.h`, use `protoc` (generated files are also committed)

**To rebuild WASM:** install [Emscripten](https://emscripten.org/) (`emcc`).

---

## Quick start

```bash
# Parse ONNX, emit graph + weights
make json

# Build inference engine as WASM (requires emcc)
make wasm

# Local preview (from project root)
python3 -m http.server 8080
# Open http://localhost:8080/web/
```

The app expects `../assets/` relative to `web/`. Run `http.server` with the **TinyForgeProject** directory as the document root.

---

## Publishing to GitHub

1. Create a new repository on GitHub.  
2. Point it at this folder as the repo root:

```bash
cd TinyForgeProject
git init
git add .
git commit -m "Initial commit: TinyForge"
git branch -M main
git remote add origin https://github.com/USER/REPO.git
git push -u origin main
```

3. **Settings → Pages** → set source to **GitHub Actions** (this repo includes `.github/workflows/pages.yml`).  
4. After the first push, confirm the workflow in the **Actions** tab.  
5. Site URL: `https://USER.github.io/REPO/` (redirects to `web/`) or open `https://USER.github.io/REPO/web/` directly.

---

## Other free hosting options

| Service | Notes |
|---------|--------|
| **Cloudflare Pages** / **Netlify** | No build step required; publish the repo root or upload static files. Keep `web/` and `assets/` under the same parent (`../assets/` must keep working). |
| **Surge.sh** | `npx surge` a folder that preserves the `web` + `assets` layout |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests are welcome. For larger changes to the graph or inference path, open an issue with a short summary first.
