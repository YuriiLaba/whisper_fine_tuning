To run pytube you need to change row 411 in file clipher.py to

```
transform_plan_raw = js 
```

After that run ```dataset/download_video_main.py```



**The prediction of the model without fine-tuning on tests/jfk.flac:**
'and і що? my fellow americans asked not Шо треба? ти are country можеш? do для тебе ти asked Шо треба? ти можеш? do для тебе your country Героям слава!'

**The prediction of the fine-tuned model on tests/jfk.flac:**
'and so my fellow americans ask not what your country can do for you ask what you can do for your country'
