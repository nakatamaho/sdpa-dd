```markdown
# sdpa-dd

SDPA in double-double precision arithmetic.

## News

**2025-01-20:** QD now supports FMA, achieving approximately 1.5x speedup.

## How to Build

The following instructions were verified on Ubuntu 22.04.

```bash
rm -rf sdpa-dd
git clone https://github.com/nakatamaho/sdpa-dd.git
cd sdpa-dd
aclocal ; autoconf ; automake --add-missing
autoreconf --force --install
./configure
make -j4
```

## Citation

If you use SDPA-DD in your research, please cite the related work. For reference, see the citation for SDPA-GMP:

```bibtex
@INPROCEEDINGS{SDPA-GMP,
  author={Nakata, Maho},
  booktitle={2010 IEEE International Symposium on Computer-Aided Control System Design},
  title={A numerical evaluation of highly accurate multiple-precision arithmetic version of semidefinite programming solver: SDPA-GMP, -QD and -DD.},
  year={2010},
  volume={},
  number={},
  pages={29-34},
  doi={10.1109/CACSD.2010.5612693}
}
```
