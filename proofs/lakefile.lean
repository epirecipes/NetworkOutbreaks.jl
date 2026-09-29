import Lake
open Lake DSL

package "NetworkOutbreaksProofs" where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

@[default_target]
lean_lib «NetworkOutbreaksProofs» where
  globs := #[.submodules `NetworkOutbreaksProofs]

