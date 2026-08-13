# Adapter Placeholder

No adapter is trained in v1.

Future extension point:

```python
class FeatureAdapter(nn.Module):
    def forward(self, features):
        return features
```

Current configs expose:

```yaml
representation_space: raw
```

A later SeeSE3-style experiment may extend this to:

```yaml
representation_space:
  - raw
  - adapted
```

The present smoke pipeline must remain unaffected by this placeholder.

