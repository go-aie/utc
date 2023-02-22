# utc

[![Go Reference](https://pkg.go.dev/badge/go-aie/utc/vulndb.svg)][2]

Go Inference API for [UTC (Universal Text Classification)][1].


## Installation

1. Install [Paddle Inference Go API][3]
2. Generate [the inference model](cli/README.md#save-inference-model)

    ```bash
    $ python3 cli/cli.py download
    ```

3. Install `utc`

    ```bash
    $ go get -u github.com/go-aie/utc
    ```


## Documentation

Check out the [documentation][2].


## Testing

Run tests:

```bash
$ go test -v -race | grep -E 'go|Test'
=== RUN   TestBuildInputsWithPrompt
--- PASS: TestBuildInputsWithPrompt (0.00s)
=== RUN   TestPromptTokenizer_Encode
--- PASS: TestPromptTokenizer_Encode (0.01s)
=== RUN   TestUTC_Run
--- PASS: TestUTC_Run (2.89s)
ok      github.com/go-aie/utc   3.748s
```


## License

[MIT](LICENSE)


[1]: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/zero_shot_text_classification
[2]: https://pkg.go.dev/github.com/go-aie/utc
[3]: https://www.paddlepaddle.org.cn/inference/master/guides/install/go_install.html
