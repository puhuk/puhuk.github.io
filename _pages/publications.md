---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

### Conference
Jinyeong Chae, Donghwa Kim, Kwanseok Kim, Doyeon Lee, **Sangho Lee**, Seongsu Ha, Jonghwan Mun, Wooyoung Kang, Byungseok Roh, Joonseok Lee. **Towards a Complete Benchmark on Video Moment Localization**, Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.

**Sangho Lee**, Seoyoung Lee, Joonseok Lee. **Learning to Wear: Details-Preserved Virtual Try-on via Disentangling Clothes and Wearer**, Proceedings of the 33rd British Machine Vision Conference (BMVC), 2022.

Jonghwan Mun, Minchul Shin, Gunsoo Han, **Sangho Lee**, Seongsu Ha, Joonseok Lee, Eun-Sol Kim. **BaSSL: Boundary-aware Self-Supervised Learning for Video Scene Segmentation**, Proceedings of the 16th Asian Conference on Computer Vision (ACCV), 2022.

### Workshop
Sangho Lee, Seoyoung Lee, Joonseok Lee. **Towards Detailed Characteristic-Preserving Virtual Try-On**, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), The 5th Workshop on Computer Vision for Fashion, Art, and Design, 2022.

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
