# Происхождение программ

Пути исходников относительны каталогам проектов автора. SHA-256 вычислен по содержимому поставляемого файла с единым переводом строк LF, чтобы сумма совпадала на Windows и Linux. Для проверки в Python используйте `hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()`. Строки «выборка функций» описывают модули, собранные из перечисленных исходников. Файлы практикума, отсутствующие в таблице, созданы для коллекции.

| Файл | Первичный источник | SHA-256 |
|---|---|---|
| [elastic/qha_core.py](elastic/qha_core.py) | `CoCrNi/temperature_elastic_20261118/payload/{common,qha}.py (выборка функций)` | `dfcb65d168f71dd43666def62657f075e8e86e777db5bc04ce57159b745cc0d7` |
| [gfa/__init__.py](gfa/__init__.py) | `GFA/src/gfa/__init__.py` | `94fa3fd4df190006473983982462ee9cb5f3c80eefbddab90fa0f328700d28d7` |
| [gfa/checks.py](gfa/checks.py) | `GFA/src/gfa/checks.py` | `d8fb85b2f78cd2f538ca80582f4ea846e2154a82cf38a11cf9a34049fba3f1f7` |
| [gfa/descriptors.py](gfa/descriptors.py) | `GFA/src/gfa/descriptors.py` | `ed61281e29abee38fe2b01191459978f228f69c88ea9ad1a55dfd349f9b1792d` |
| [gfa/dynamics.py](gfa/dynamics.py) | `GFA/src/gfa/dynamics.py` | `3345da8f9c2b7c6d34d63eab385e947cae67720bfd8756c33e06b1873aa144ab` |
| [gfa/equilibration.py](gfa/equilibration.py) | `GFA/src/gfa/equilibration.py` | `2f4125df2ad89cb29a5dc1c18ac9d1d645a6d8192e673257d25f895a6a1420b1` |
| [gfa/interface_screening.py](gfa/interface_screening.py) | `GFA/src/gfa/interface_screening.py` | `e69c90513dbaafdeb5169b5e83d6865629656e8a0dc92fc37074e094d8449d0b` |
| [gfa/interface_series.py](gfa/interface_series.py) | `GFA/src/gfa/interface_series.py` | `ed10085a4de444ece4ed4848dc9f36548a47873833dfa8a7319b66656156c78e` |
| [gfa/model_comparison.py](gfa/model_comparison.py) | `GFA/src/gfa/model_comparison.py` | `2f9fdb03ac3a408c26669c06b9ee57f1040177af42211522b145dd857b11cb3b` |
| [gfa/motif_lifetimes.py](gfa/motif_lifetimes.py) | `GFA/src/gfa/motif_lifetimes.py` | `849b650339741dcb2fd3f7b4b358ece3b8040cfcf07a2c77e937214a4cf5cfb1` |
| [gfa/neighbor_dynamics.py](gfa/neighbor_dynamics.py) | `GFA/src/gfa/neighbor_dynamics.py` | `15de710942a0d03dd675135f7e1eb2db13343afe4ab2cfc807c6eba0ff8d9772` |
| [gfa/structural_atlas.py](gfa/structural_atlas.py) | `GFA/src/gfa/structural_atlas.py` | `b47593b2bdb4e6c4b5db20956abd333af7ef8d72dc5f0191ed2c40607a635eee` |
| [oxide/composition_curve.py](oxide/composition_curve.py) | `Viscosity/scripts/composition_curve.py` | `f9b1ccb14a16eb85bb9b5c13f3ef2bf0d9b3223da5e5e763d9a5e3dae3fd59a8` |
| [oxide/oxide_basis.py](oxide/oxide_basis.py) | `Viscosity/scripts/oxide_basis.py` | `d3bc5f4b424608ce500638a17bed49c0778a05b1e043da6fc9a551eedb47f2c6` |
| [screening/__init__.py](screening/__init__.py) | `ML-Materials/codex_screening_v2/pilot/__init__.py` | `64c7bace491972c91b78e1fd7c23b5994f8080ed816680533fef25beebf59f71` |
| [screening/domain.py](screening/domain.py) | `ML-Materials/codex_screening_v2/pilot/domain.py` | `ee15cd2b364f06d5996d6d330b4f380180322fbc8fc26262fe34abbeec6f58d9` |
| [screening/expressions.py](screening/expressions.py) | `ML-Materials/codex_screening_v2/pilot/expressions.py` | `3d53c2b53dd85e31684ceea4d5c961856c047477c71acd25875d583fe5baaf87` |
| [screening/normalize.py](screening/normalize.py) | `ML-Materials/codex_screening_v2/pilot/normalize.py` | `228b874337b5808ff564a215226af105c7cc3bf3fcad54f9dac011329bb6ecf0` |
| [screening/selection.py](screening/selection.py) | `ML-Materials/codex_screening_v2/pilot/selection.py` | `b82af529157d76b7cda364eea352a24ae77330fcf0be100d6ec0829fe1123346` |
| [symbolic/formula_tools.py](symbolic/formula_tools.py) | `ТПР-2026/solver/public/formula_tools.py` | `838b8fa20727886c2e48f8d45b03fe90139c2099fc5a919eebe5f810b82f467f` |
| [tests/test_descriptors.py](tests/test_descriptors.py) | `GFA/tests/test_descriptors.py` | `86de5833c0783ba0326f7add3ee0f0356cc8dac213c304fdafb827ab71fff1f4` |
| [tests/test_dynamics.py](tests/test_dynamics.py) | `GFA/tests/test_dynamics.py` | `a9323518f8f9d5b7d466b41cf93b4a75233249e0be6a82ce1c0a51759c2640bc` |
| [tests/test_equilibration.py](tests/test_equilibration.py) | `GFA/tests/test_equilibration.py` | `afc2359cbc0e6457f17bab77114230d008f691f49054145b927724a75914bb30` |
| [tests/test_interface_screening.py](tests/test_interface_screening.py) | `GFA/tests/test_interface_screening.py` | `1a74af41cffde63461683abd7d56097eb15dfebfd7d70e1991bbea8eb72c1326` |
| [tests/test_interface_series.py](tests/test_interface_series.py) | `GFA/tests/test_interface_series.py` | `928a810dc960fecc7a76cb2f160af56e798a32e41b45c782132f7204e41c78ed` |
| [tests/test_model_comparison.py](tests/test_model_comparison.py) | `GFA/tests/test_model_comparison.py` | `8e7d5426de103c1fe92bbda756a7ce5c2928b665eaf1a93da4b2fd604d0c2ffb` |
| [tests/test_motif_lifetimes.py](tests/test_motif_lifetimes.py) | `GFA/tests/test_motif_lifetimes.py` | `23a1ba787473b8940b86306932fce290c241df934f4637c0180dc44942ca963b` |
| [tests/test_neighbor_dynamics.py](tests/test_neighbor_dynamics.py) | `GFA/tests/test_neighbor_dynamics.py` | `4594401ad88828a5a1725f08442c411e0e4bea759311f5a99d0b5b85b0aa8e01` |
| [symbolic/native_tree.py](symbolic/native_tree.py) | `sr-pilot-v2-1/solver_handoff/v2/pilot_results/drivers/native_backend.py` (выборка функций) | `6db75ba213c1df1f5b6557879b251edadb51fe6afa6135bd3c304897a7e8a38c` |
| [screening/nn_replay.py](screening/nn_replay.py) | `blind-validation-r02-v3/public_project/pilot/models.py` (выборка функций) | `063f6ebc5dba4ddd1f7c720243d1d717d40d888f0949fc29ba77532502d402b6` |
| [oxide/study_weights.py](oxide/study_weights.py) | `Viscosity/bof_transfer/transfer.py` (выборка функций) | `cc908e0e015733137be91e112b7183da7109cca403af18440b630cde8371f4d0` |
| [oxide/audit_dataset.py](oxide/audit_dataset.py) | `Viscosity/scripts/audit_dataset.py` | `27638159e0905139a42f749e84b61ba51f36a902b349e7643696a2bd8cfa41f4` |
| [oxide/classical_benchmarks.py](oxide/classical_benchmarks.py) | `Viscosity/scripts/classical_benchmarks.py` | `5256c52a870327fd167807af2cfc4015134f50ae9186c036b16f85773b24a62d` |
| [oxide/compare_models.py](oxide/compare_models.py) | `Viscosity/scripts/compare_models.py` | `94c0e78fb617bea8c039ed1c9944f6c91f139d0b484ce0595412afd1777e2ba8` |
| [oxide/iida_transfer.py](oxide/iida_transfer.py) | `Viscosity/scripts/iida_transfer.py` | `01967017f9eddc8eeb0713aa8e262c409c584e6d02c21d9659ea6196d95cb301` |
| [oxide/plot_benchmark.py](oxide/plot_benchmark.py) | `Viscosity/scripts/plot_benchmark.py` | `43e890157d42c38238e302386542422214baf231c27bfd81aab94784d7c714ac` |
| [oxide/predict_viscosity.py](oxide/predict_viscosity.py) | `Viscosity/scripts/predict_viscosity.py` | `9379e613ba0b8891e4f16a4e767609e2cc363ce706a658c152a628fa763e2942` |
| [oxide/reproduce_gpvisc.py](oxide/reproduce_gpvisc.py) | `Viscosity/scripts/reproduce_gpvisc.py` | `2d3f1ea1787eb42f54aee6a8977ef5c953dce1a34c481bb263966c6d8216de0e` |
| [oxide/verify_gpvisc.py](oxide/verify_gpvisc.py) | `Viscosity/scripts/verify_gpvisc.py` | `442b36498d9987241b4a8067eea8a21aabff9c36bfd9caa088bb4672f1b428fd` |
| [oxide/verify_inference.py](oxide/verify_inference.py) | `Viscosity/scripts/verify_inference.py` | `7fb4c163ab0365c6f0423951de14770f5810f2ea66bf5e1a26245980c3f203dd` |
| [oxide/requirements-benchmark.txt](oxide/requirements-benchmark.txt) | `Viscosity/requirements-benchmark.txt` | `82914444f2355f16a93eea7353cd7b109f8ba54a60deb787223ac5d90a4194b3` |
| [oxide/requirements-gpvisc.txt](oxide/requirements-gpvisc.txt) | `Viscosity/requirements-gpvisc.txt` | `8142a6ee8846ceba3baa1fdf5814e47b46353ed4a21d0eb02aca938e55798b13` |
