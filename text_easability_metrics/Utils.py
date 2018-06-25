def transform_csv_to_structure(df):
    text_ids = df['id'].unique()
    texts = list()
    for text_id in text_ids:
        sentence_grouped = df.where(dfSeventh['id'] == 1)
        clean_df = df[pd.notnull(sentence_grouped['id'])]
        paragraphs = list()
        df_paragraphs = clean_df.groupby('paragraph_id').apply(lambda x: "%s" % '|'.join(x['value'])).values
        for df_paragraph in df_paragraphs:
            paragraphs.append(df_paragraph.split('|'))
        texts.append(paragraphs)

    return texts