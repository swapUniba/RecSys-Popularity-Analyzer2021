import fastparquet

pfile = fastparquet.ParquetFile('../recs/cf/runs.parquet')
with open ('runs.csv', 'w') as fp:
    for i, df in enumerate(pfile.iter_row_groups()):
        write_header = (i==0)
        df.to_csv(fp, index=False, header=write_header)
