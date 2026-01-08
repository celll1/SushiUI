# DEUS推論パフォーマンス分析レポート

## 問題
SDXLよりサイズが小さいにも関わらず、DEUSの推論が2.5倍遅い。

## 特定されたボトルネック

### 1. **RoPE2Dの毎回計算（最重要）**

**問題**: `RoPE2D.forward()`が毎回以下の計算を実行している：
- `torch.arange(H, W)` - 位置インデックスの生成
- `torch.einsum` - 周波数計算（2回）
- `torch.sin/cos` - 三角関数計算（4回）
- `unsqueeze/expand` - メモリ展開操作
- `permute` - メモリ転置

**影響**: 各推論ステップで、U-NetのforwardごとにRoPE計算が実行される。
- 解像度108x192の場合: H*W = 20,736ピクセル
- 各ステップで20,736回のsin/cos計算 + einsum演算

**DiffusersのSDXL**: 位置エンコーディングは事前計算またはキャッシュされる。

**推奨修正**:
```python
# キャッシュ実装
self._cache = {}  # {(H, W): cached_embedding}

def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    cache_key = (H, W)
    
    if cache_key in self._cache:
        emb_2d = self._cache[cache_key].to(x.device, dtype=x.dtype)
    else:
        # 計算（初回のみ）
        emb_2d = self._compute_rope(H, W, x.device, x.dtype)
        self._cache[cache_key] = emb_2d.cpu()  # CPUにキャッシュ
    
    return x + emb_2d
```

**期待される改善**: 10-20%の速度向上

---

### 2. **Attentionの過剰なreshape/permute操作**

**問題**: `CrossAttentionBlock.forward()`で以下の操作が頻繁に実行される：
```python
# [B, C, H, W] -> [B, H*W, C]
x_flat = x.view(B, C, H * W).permute(0, 2, 1).contiguous()

# QKV計算後に再度reshape
q = self.to_q(x_norm).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
# ... (同様の操作が6回)

# 戻す
x = x_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)
```

**影響**: 
- `permute`と`contiguous()`はメモリコピーを引き起こす
- 各attentionブロックで6-8回のreshape操作
- メモリ帯域幅の無駄

**DiffusersのSDXL**: Attention processorを使用し、reshapeを最小化。

**推奨修正**:
- Attention processorパターンを採用
- または、reshapeを最小限に抑えた実装

**期待される改善**: 5-10%の速度向上

---

### 3. **torch.compile未使用**

**問題**: DEUS U-Netに`torch.compile`が適用されていない。

**DiffusersのSDXL**: 多くの場合、`torch.compile`が推奨される。

**推奨修正**:
```python
# pipeline.pyまたはmodel_loader.pyで
if use_torch_compile:
    unet = torch.compile(unet, mode="reduce-overhead")
```

**期待される改善**: 20-30%の速度向上（初回実行後）

---

### 4. **FFNの実装**

**問題**: FFNが各attentionブロックで実行されているが、最適化されていない可能性。

**確認が必要**: DiffusersのSDXLと比較して、FFNの実装が効率的か。

---

### 5. **メモリ転送の最適化不足**

**問題**: 
- `contiguous()`の呼び出しが多すぎる
- 不要なdtype変換がある可能性

**推奨修正**:
- `contiguous()`を必要最小限に
- dtype変換を事前にまとめる

---

## 優先順位

1. **最優先**: RoPE2Dのキャッシュ化（実装が簡単で効果大）
2. **高優先**: torch.compileの適用
3. **中優先**: Attentionのreshape最適化
4. **低優先**: その他の細かい最適化

## 期待される総合改善

- RoPE2Dキャッシュ: +10-20%
- torch.compile: +20-30%
- Attention最適化: +5-10%

**合計**: 約1.4-1.7倍の速度向上が期待される（2.5倍遅い → 1.5-1.8倍遅い程度まで改善）

## 実装完了

✅ **1. RoPE2Dキャッシュ化**: 実装完了
- 解像度ごとにRoPE埋め込みをキャッシュ
- CPUにキャッシュしてVRAMを節約
- 期待される改善: +10-20%

✅ **2. torch.compile適用**: 実装完了
- `use_torch_compile`パラメータで有効化可能
- `reduce-overhead`モードで最適化
- 期待される改善: +20-30%

✅ **3. Attention最適化**: 実装完了
- 不要な`contiguous()`呼び出しを削減
- `reshape`を`view`の代わりに使用（より柔軟）
- 期待される改善: +5-10%

## 使用方法

### torch.compileを有効化

生成パラメータに`use_torch_compile: true`を追加：

```python
params = {
    "prompt": "1girl",
    "use_torch_compile": True,  # これを追加
    # ... 他のパラメータ
}
```

### RoPEキャッシュの確認

```python
# U-NetのRoPEキャッシュサイズを確認
cache_size = unet.rope_2d.get_cache_size()
print(f"RoPE cache entries: {cache_size}")

# 必要に応じてキャッシュをクリア
unet.rope_2d.clear_cache()
```

## 次のステップ

1. ✅ RoPE2Dキャッシュを実装 - 完了
2. ✅ torch.compileを有効化 - 完了
3. ✅ Attention実装を最適化 - 完了
4. プロファイリングで実際の改善を測定
5. 必要に応じて追加の最適化を検討
