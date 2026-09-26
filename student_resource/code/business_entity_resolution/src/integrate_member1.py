"""Adapter to run Member 1 blocking and produce candidate_pairs.tsv for Member 2."""
import os
import sys
import importlib.util
import pandas as pd

# ensure member1 repo is importable without collisions
MEMBER1_REPO = "/home/piyush/VS/AmazonMLChallenge/member1_repo"
if not os.path.exists(MEMBER1_REPO):
    MEMBER1_REPO = "/home/piyush/VS/AmazonMLChallenge/student_resource/member1_repo"
m1_src_dir = os.path.join(MEMBER1_REPO, "src")

# Load member1's normalize and candidate_generation directly
norm_spec = importlib.util.spec_from_file_location("src.preprocessing.normalize", os.path.join(m1_src_dir, "preprocessing", "normalize.py"))
norm_mod = importlib.util.module_from_spec(norm_spec)
sys.modules["src.preprocessing"] = importlib.util.module_from_spec(importlib.util.spec_from_file_location("src.preprocessing", os.path.join(m1_src_dir, "preprocessing", "__init__.py")))
sys.modules["src.preprocessing.normalize"] = norm_mod
norm_spec.loader.exec_module(norm_mod)

cg_spec = importlib.util.spec_from_file_location("src.blocking.candidate_generation", os.path.join(m1_src_dir, "blocking", "candidate_generation.py"))
m1_cg = importlib.util.module_from_spec(cg_spec)
sys.modules["src.blocking"] = importlib.util.module_from_spec(importlib.util.spec_from_file_location("src.blocking", os.path.join(m1_src_dir, "blocking", "__init__.py")))
sys.modules["src.blocking.candidate_generation"] = m1_cg
cg_spec.loader.exec_module(m1_cg)


from .config import cfg


def build_candidates_for_dir(s1_path, s2_path, s3_path, out_path):
    print(f"Loading sources for candidate generation from {os.path.dirname(s1_path)}...")
    s1 = m1_cg.load_source(s1_path)
    s2 = m1_cg.load_source(s2_path)
    s3 = m1_cg.load_source(s3_path)

    print(f"Generating S1->S2 candidates ({len(s1)} S1, {len(s2)} S2)...")
    c2 = m1_cg.generate_candidates(s1, s2)
    print(f"Generating S1->S3 candidates ({len(s1)} S1, {len(s3)} S3)...")
    c3 = m1_cg.generate_candidates(s1, s3)

    combined = pd.concat([c2, c3], ignore_index=True)
    grouped = combined.groupby("source1_entity_id")["matched_entity_id"].agg(lambda s: ",".join(sorted(set(s)))).reset_index()

    # ensure all S1 present
    all_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)["entity_id"].tolist()
    out_df = pd.DataFrame({"source1_entity_id": all_s1}).merge(grouped, how="left", on="source1_entity_id")
    out_df = out_df.rename(columns={"matched_entity_id": "candidate_entity_ids"})
    out_df["candidate_entity_ids"] = out_df["candidate_entity_ids"].fillna("")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote candidates to {out_path}, rows={len(out_df)}")


def main():
    test_dir = str(cfg.test_dir)
    train_dir = str(cfg.train_dir)
    out_dir = str(cfg.output_dir)

    # train candidates
    train_cand_out = os.path.join(out_dir, "train_candidate_pairs.tsv")
    print("\n--- Generating Train Candidates ---")
    build_candidates_for_dir(
        os.path.join(train_dir, "train_source1.tsv"),
        os.path.join(train_dir, "train_source2.tsv"),
        os.path.join(train_dir, "train_source3.tsv"),
        train_cand_out,
    )

    # test candidates
    test_cand_out = os.path.join(out_dir, "candidate_pairs.tsv")
    print("\n--- Generating Test Candidates ---")
    build_candidates_for_dir(
        os.path.join(test_dir, "test_source1.tsv"),
        os.path.join(test_dir, "test_source2.tsv"),
        os.path.join(test_dir, "test_source3.tsv"),
        test_cand_out,
    )


if __name__ == "__main__":
    main()

