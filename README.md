This repo is the official implementation of 
“Make Privacy Renewable! Generating Privacy-Preserving Faces Supporting Cancelable Biometric Recognition”  (2024 ACM MM)

Tao Wang, Yushu Zhang, Xiangli Xiao, Lin Yuan, Zhihua Xia, Jian Weng

预训练模型：链接：https://pan.baidu.com/s/1Lx9Lvq8bIBO8npRrmzWZZA?pwd=ojf7 
提取码：ojf7

对于输入的人脸（训练或测试），需要使用人脸对齐处理代码将其设定为128像素

![image](Teaser_Image.png)

Step one: training the Auxiliary Physical Identity Remover

run training_for_Remover.py,   You can adjust the parameters lambda_rec and lambda_id


Step two: training  CanFG

run  training_for_CanFG.py,  You can adjust the parameters lambda_rec, lambda_em and lp, and modify the seed in CanFG.py "torch.manual_seed(85)" to obtain other random_orthogonal_matrix.



## License

更多的评估指标或数据可以参考 https://github.com/fkeufss/PRO-Face

```
@inproceedings{wang2024make,
  title={Make Privacy Renewable! Generating Privacy-Preserving Faces Supporting Cancelable Biometric Recognition},
  author={Wang, Tao and Zhang, Yushu and Xiao, Xiangli and Yuan, Lin and Xia, Zhihua and Weng, Jian},
  booktitle={Proc.  ACM Int. Conf. Multimedia},
  pages={10268--10276},
  year={2024},
  doi={10.1145/3664647.3680704}
}
```
