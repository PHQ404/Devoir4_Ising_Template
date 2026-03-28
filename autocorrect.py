import os
from typing import Optional

import tac


def get_report_from_url(
    repo_url: Optional[str] = None,
    master_repo_url: Optional[str] = None,
    path_to_root=".",
    weights=None,
):
    code_source = tac.SourceCode(
        os.path.join(path_to_root, "src"), url=repo_url, logging_func=print
    )
    tests_source = tac.SourceTests(
        os.path.join(path_to_root, "tests"), url=repo_url, logging_func=print
    )
    if master_repo_url is None:
        master_code_source, master_tests_source = None, None
    else:
        master_code_source = tac.SourceMasterCode(
            os.path.join(path_to_root, "src"),
            url=master_repo_url,
            logging_func=print,
            local_repo_tmp_dirname="tmp_master_repo",
        )
        master_tests_source = tac.SourceMasterTests(
            os.path.join(path_to_root, "tests"),
            url=master_repo_url,
            logging_func=print,
            local_repo_tmp_dirname="tmp_master_repo",
        )
    default_weights = {
        tac.Tester.PEP8_KEY: 10.0,
        tac.Tester.PERCENT_PASSED_KEY: 15.0,
        tac.Tester.CODE_COVERAGE_KEY: 10.0,
        tac.Tester.MASTER_PERCENT_PASSED_KEY: 15.0,
    }
    if weights is None:
        weights = {}
    weights = {**default_weights, **weights}

    auto_corrector = tac.Tester(
        code_source,
        tests_source,
        master_code_src=master_code_source,
        master_tests_src=master_tests_source,
        report_dir="tmp_report_dir",
        logging_func=print,
        weights=weights,
    )
    auto_corrector.run(
        overwrite=False,
        debug=True,
        clear_temporary_files=False,
        clear_pytest_temporary_files=False,
    )
    auto_corrector.rm_report_dir()
    return auto_corrector.report


def get_grade_report():
    template_url = r"https://github.com/PHQ404/Devoir4_Ising_Template.git"
    base_grade = get_report_from_url(
        repo_url=template_url, master_repo_url=template_url
    ).grade
    report = get_report_from_url(repo_url=None, master_repo_url=template_url)
    new_report = tac.Report(
        data=report.data,
        grade_min=base_grade,
        grade_min_value=15.0,
        grade_max=100.0,
        report_filepath=os.path.join(os.path.dirname(__file__), "grade_report.json"),
    )
    print(new_report)
    new_report.save()
    return new_report


if __name__ == "__main__":
    get_grade_report()
