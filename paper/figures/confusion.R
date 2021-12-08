box::use(
    here[here],
    readr[read_csv],
    dplyr[select, full_join, mutate],
    tibble[as_tibble],
    ggplot2[...],
    cvms[plot_confusion_matrix, evaluate]
)

test_results <- read_csv(here("data/out/test_results.csv"), show_col_types = FALSE)
conf_model <- test_results |>
    select(-c(rule, text)) |>
    table() |>
    as_tibble() |>
    mutate(pred = as.numeric(pred), truth = as.numeric(truth))

model_eval <- evaluate(
    test_results |> select(-c(rule, text)),
    target_col = "truth",
    prediction_col = "pred",
    type = "binomial"
)

rule_eval <- evaluate(
    test_results |> select(-c(pred, text)),
    target_col = "truth",
    prediction_col = "rule",
    type = "binomial"
)

eval_table <- model_eval |>
    full_join(rule_eval) |>
    as_tibble() |>
    select(c(
        Accuracy,
        F1,
        Sensitivity,
        Specificity
    ))

conf_model_plot <- plot_confusion_matrix(
    conf_model,
    target_col = "truth", prediction_col = "pred", counts_col = "n",
    tile_border_color = "black",
    tile_border_size = .5,
    palette = "Greys"
)

conf_rule <- test_results |>
    select(-c(pred, text)) |>
    table() |>
    as_tibble() |>
    mutate()

conf_rule_plot <- plot_confusion_matrix(
    conf_rule,
    target_col = "truth", prediction_col = "rule", counts_col = "n",
    tile_border_color = "black",
    tile_border_size = .5,
    palette = "Greys"
)
