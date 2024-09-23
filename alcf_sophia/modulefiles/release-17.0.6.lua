--[[

    file llvm module
]]--

local root_path = "/soft/compilers/clang/17.0.6"

prepend_path("PATH", pathJoin(root_path, "bin"), ":")
prepend_path("PATH",  pathJoin(root_path, "share/clang"), ":")
prepend_path("MANPATH", pathJoin(root_path, "share/man"), ":")
prepend_path("LD_LIBRARY_PATH", pathJoin(root_path, "lib"), ":")

-- KGF: uncommented on Polaris modulefile:
-- setenv("LIBOMP_USE_HIDDEN_HELPER_TASK", "0")
-- setenv("LIBOMPTARGET_MAP_FORCE_ATOMIC", "FALSE")
