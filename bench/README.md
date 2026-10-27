# Experimental real-time benchmark

These scripts explore single-frame P2VM reduction and timing. They are not part
of the installed pipeline. Produce the three calibration FITS products first,
then set the calibration directory and raw data paths near their use in
`bench/rtreduce.py`. The benchmark reads `planetes_flat.fits`,
`planetes_wave.fits`, and `planetes_p2vm.fits`; it does not load runtime YAML.

```bash
python bench/rtreduce.py
```

The benchmark's numerical operations and timing loop are unchanged. It is not
an ESO recipe or a substitute for the full-cube science reduction.
