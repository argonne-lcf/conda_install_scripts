import sys, os, faulthandler; faulthandler.enable()
v = sys.argv[1]
def tri():
    import torch, triton
    f = torch.compile(lambda x: (x*2).sin()); return f"compile {f(torch.randn(64,device='cuda')).shape} triton {triton.__version__}"
def tf(): import tensorflow as tf; return tf.reduce_sum(tf.ones((8,8))).numpy()
def jx(): import jax, jax.numpy as jnp; return float(jnp.ones(8).sum())
if v == "tf_then_triton": tf(); print("OK", v, tri())
elif v == "jax_then_triton": jx(); print("OK", v, tri())
elif v == "triton_then_tf": r=tri(); tf(); print("OK", v, r)
elif v == "tf_then_triton_deepbind":
    tf(); sys.setdlopenflags(os.RTLD_NOW|os.RTLD_DEEPBIND); print("OK", v, tri())
elif v == "tf_local_then_triton":
    # force TF's ImportError path -> no pywrap_dlopen_global_flags -> RTLD_LOCAL
    sys.modules['tensorflow.python.pywrap_dlopen_global_flags'] = None
    tf(); print("OK", v, tri())
elif v == "tf_local_then_fa":
    sys.modules['tensorflow.python.pywrap_dlopen_global_flags'] = None
    tf(); jx()
    import torch; from flash_attn import flash_attn_func
    q = torch.randn(1,128,8,64, device="cuda", dtype=torch.float16); o = flash_attn_func(q,q,q); torch.cuda.synchronize()
    print("OK", v, tuple(o.shape))
