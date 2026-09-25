# data/ (não versionado)

As imagens das lâminas não sobem para o git (repositório público; dados do projeto).

Coloque aqui, em `data/patches/`, os patches exportados pelo `rockface`:

```
data/patches/patch_y22800_x34200_c0.png
data/patches/patch_y22800_x38000_c0.png
...
```

- `patch_y{Y}_x{X}_c0.png`: canal 0 (luz normal/PPL), RGB 8 bits, 4096 × 4096 px,
  passo de patching 3800 px.
- Conjunto atual: 18 patches de uma lâmina (≈ 280 MB), obtidos com a equipe.

Para usar outra pasta sem copiar os arquivos: `LAMINAS_DATA=<pasta>` (ver
`docs/GUIA_DE_USO.md`, seção 3).
