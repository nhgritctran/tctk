import hail as hl
from tqdm import tqdm


def scan_interval(locus, radius=10):
    """
    generate variant scan interval around target site
    :param locus: target locus
    :param radius: number of bases away from target locus
    :return: scan interval that can be used by Hail
    """

    c = locus.split(":")[0]
    pos = int(locus.split(":")[1])
    si = c + ":" + str(pos - radius) + "-" + str(pos + radius)

    return si


def find_variant(mt, var_dict, snv=True):
    """
    take a variant dict and scan input matrix table for these variants
    :param mt: Hail matrix table
    :param var_dict: variant dict; example:
    {'SMAD2': ['chr18:47848564:G:C',
               'chr18:47841885:A:G'],
     'SMAD3': ['chr15:67165385:G:C',
               'chr15:67066361:G:C'],
    :param snv: boolean; determine type variant for appropriate mapping approach
    :return: matched variant dict and matched matrix table
    """

    # check all ClinVar variants in AoU data and return variants detected and their matrix tables
    variants = {}
    final_mt = {}

    for k, v in tqdm(var_dict.items()):
        print()
        print("==========")
        print()
        print(k)
        variants[k] = []
        for i, var in enumerate(tqdm(v)):
            parsed_var = hl.parse_variant(var, reference_genome="GRCh38")
            hl.eval(parsed_var)

            locus = var.split(":")[0] + ":" + var.split(":")[1]
            filtered_mt = hl.filter_intervals(mt,
                                              [hl.parse_locus_interval(x, reference_genome="GRCh38")
                                               for x in [scan_interval(locus)]])

            # split multi-allelic sites
            bi = filtered_mt.filter_rows(hl.len(filtered_mt.alleles) == 2)
            bi = bi.annotate_rows(a_index=1, was_split=False)
            multi = filtered_mt.filter_rows(hl.len(filtered_mt.alleles) > 2)
            split = hl.split_multi_hts(multi)
            region_to_scan = split.union_rows(bi)

            print()
            print("target variant with approximate genomic position: ", var)
            print()
            print("variants in scan region:")
            region_to_scan.row.show()

            if snv:
                # for SNVs: match both position and nucleotide change
                temp_mt = region_to_scan.filter_rows((region_to_scan["locus"] == parsed_var["locus"]) &
                                                     (region_to_scan["alleles"][0] == parsed_var["alleles"][0]) &
                                                     (region_to_scan["alleles"][1:].contains(
                                                         parsed_var["alleles"][1])))

            else:
                # for indels: filter mt to keep only a region and check for nucleotide change
                temp_mt = region_to_scan.filter_rows((region_to_scan["alleles"][0] == parsed_var["alleles"][0]) &
                                                     (region_to_scan["alleles"][1:].contains(
                                                         parsed_var["alleles"][1])))

            c = temp_mt.count()[0]
            if c:
                print()
                print("FOUND MATCHED VARIANT!!!")
                temp_mt.row.show()
                variants[k].append(var)
                try:
                    final_mt[k] = final_mt[k].union_rows(temp_mt)
                except Exception as e:
                    print(e)
                    final_mt[k] = temp_mt

            print()
            print("----------")
            print()

    return variants, final_mt
