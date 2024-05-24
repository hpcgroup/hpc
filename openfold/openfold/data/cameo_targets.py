# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following data comes from:
# https://cameo3d.org/modeling/targets/3-months/?to_date=2021-12-11
# CAMEO target id format is mostly "pdb_id [mmcif_chain_id]"
# and occasionally "pdb_id [author_chain_id]".

CAMEO_TARGETS = {
    "2021-09-18": [
        "7bgg [A]",
        "7bly [A]",
        "7cz9 [B]",
        "7d2w [A]",
        "7e6g [A]",
        "7jta [A]",
        "7jua [A]",
        "7ksn [A]",
        "7lm0 [A]",
        "7m1w [A]",
        "7o86 [A]",
        "7rih [B]",
        "7rij [A]",
        "7s3u [A]",
        "7s5n [B]",
    ],
    "2021-09-25": [
        "7ba4 [A]",
        "7czo [A]",
        "7dg7 [A]",
        "7dg9 [A]",
        "7dlw [B]",
        "7du6 [A]",
        "7du7 [A]",
        "7dvr [A]",
        "7dxs [C]",
        "7dxv [B]",
        "7dxw [A]",
        "7e24 [D]",
        "7e57 [C]",
        "7k3s [A]",
        "7ltc [A]",
        "7mbf [A]",
        "7mj0 [C]",
    ],
    "2021-10-02": [
        "7a59 [B]",
        "7a73 [A]",
        "7aao [A]",
        "7agj [A]",
        "7aj6 [L]",
        "7bmu [A]",
        "7bsx [A]",
        "7d3y [A]",
        "7d50 [B]",
        "7e4m [A]",
        "7jyp [A]",
        "7n79 [A]",
        "7ny0 [A]",
        "7og3 [A]",
        "7p01 [A]",
        "7sbc [C]",
    ],
    "2021-10-09": [
        "7d6l [A]",
        "7d8r [A]",
        "7d8s [A]",
        "7jrk [A]",
        "7p2f [A]",
        "7pmu [A]",
        "7r65 [A]",
        "7rk1 [C]",
        "7rk2 [C]",
        "7s6e [B]",
    ],
    "2021-10-16": [
        "7amc [A]",
        "7da9 [A]",
        "7dah [E]",
        "7dnm [B]",
        "7eea [A]",
        "7eft [B]",
        "7kgc [A]",
        "7nmb [A]",
        "7o62 [B]",
        "7p1b [C]",
        "7p1v [A]",
        "7rrm [C]",
        "7s13 [C]",
        "7s13 [L]",
        "7sh3 [A]",
    ],
    "2021-10-23": [
        "7dck [A]",
        "7f17 [B]",
        "7msj [A]",
        "7nwz [D]",
        "7nx0 [B]",
        "7nx0 [E]",
        "7og0 [B]",
        "7pkx [A]",
        "7plq [B]",
        "7s6g [A]",
        "7siq [A]",
    ],
    "2021-10-30": [
        "7ar0 [B]",
        "7cjs [B]",
        "7dcm [A]",
        "7ef6 [A]",
        "7et8 [A]",
        "7lc5 [A]",
        "7ndr [A]",
        "7nl4 [A]",
        "7o49 [F]",
        "7s6b [A]",
        "7sir [A]",
        "7v1v [A]",
        "7vmu [A]",
    ],
    "2021-11-06": [
        "6wmk [A]",
        "6wqc [A]",
        "7au7 [A]",
        "7b3a [A]",
        "7bcz [A]",
        "7bhy [A]",
        "7dru [C]",
        "7eqx [A]",
        "7eqx [C]",
        "7esx [A]",
        "7f7n [A]",
        "7fbp [B]",
        "7fe3 [A]",
        "7lew [B]",
        "7mfw [B]",
        "7obm [A]",
        "7ool [A]",
        "7r84 [A]",
        "7rwk [A]",
    ],
    "2021-11-13": [
        "6xqj [A]",
        "6z01 [A]",
        "7atr [A]",
        "7dfe [A]",
        "7dtp [A]",
        "7kik [A]",
        "7kos [A]",
        "7ouq [A]",
        "7p82 [C]",
        "7pbk [A]",
        "7q47 [A]",
        "7re4 [A]",
        "7re6 [A]",
    ],
    "2021-11-20": [
        "6tf4 [A]",
        "7awk [A]",
        "7bcj [A]",
        "7djy [A]",
        "7dk9 [A]",
        "7dvn [A]",
        "7kqv [D]",
        "7ls0 [B]",
        "7ny6 [A]",
        "7plb [B]",
        "7rbw [A]",
        "7vnb [A]",
    ],
    "2021-11-27": [
        "7b0d [A]",
        "7e2v [A]",
        "7f6e [B]",
        "7fbh [A]",
        "7fh3 [A]",
        "7kiu [A]",
        "7kua [A]",
        "7l6j [A]",
        "7l6y [A]",
        "7lxc [A]",
        "7swh [A]",
        "7swk [B]",
    ],
    "2021-12-04": [
        "7b1k [B]",
        "7b1w [F]",
        "7b26 [C]",
        "7b28 [F]",
        "7b29 [A]",
        "7b2a [A]",
        "7b2o [A]",
        "7bny [B]",
        "7dkk [A]",
        "7dko [A]",
        "7dmf [A]",
        "7dms [A]",
        "7lbu [A]",
        "7ljh [A]",
        "7poh [B]",
        "7ppp [A]",
        "7q03 [A]",
    ],
    "2021-12-11": [
        "7b4q [A]",
        "7dnu [A]",
        "7e37 [A]",
        "7kuw [A]",
        "7kzh [A]",
        "7mfi [A]",
        "7p3t [B]",
        "7q1b [A]",
        "7sf6 [A]",
        "7sy9 [A]",
        "7t24 [A]",
        "7v5y [B]",
        "7vdy [A]",
    ],
}