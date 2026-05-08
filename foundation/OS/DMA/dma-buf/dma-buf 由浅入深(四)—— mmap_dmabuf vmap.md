    <!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="utf-8">
    <link rel="canonical" href="https://blog.csdn.net/hexiaolong2009/article/details/102596791"/>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <meta name="renderer" content="webkit"/>
    <meta name="force-rendering" content="webkit"/>
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="report" content='{"pid": "blog", "spm":"1001.2101"}'>
    <meta name="referrer" content="always">
    <meta http-equiv="Cache-Control" content="no-siteapp" /><link rel="alternate" media="handheld" href="#" />
    <meta name="shenma-site-verification" content="5a59773ab8077d4a62bf469ab966a63b_1497598848">
    <meta name="applicable-device" content="pc">
    <link  href="https://g.csdnimg.cn/static/logo/favicon32.ico"  rel="shortcut icon" type="image/x-icon" />
    <title>dma-buf 由浅入深（四） —— mmap_dmabuf vmap-CSDN博客</title>
    <script>
      (function(){ 
        var el = document.createElement("script"); 
        el.src = "https://s3a.pstatp.com/toutiao/push.js?1abfa13dfe74d72d41d83c86d240de427e7cac50c51ead53b2e79d40c7952a23ed7716d05b4a0f683a653eab3e214672511de2457e74e99286eb2c33f4428830"; 
        el.id = "ttzz"; 
        var s = document.getElementsByTagName("script")[0]; 
        s.parentNode.insertBefore(el, s);
      })(window)
    </script>
        <meta name="keywords" content="dmabuf vmap">
        <meta name="csdn-baidu-search"  content='{"autorun":true,"install":true,"keyword":"dmabuf vmap"}'>
    <meta name="description" content="文章浏览阅读2.7w次，点赞24次，收藏54次。本文深入探讨了dma-buf在用户空间的访问方法，包括直接使用fd进行mmap操作和利用dma_buf_mmap简化设备驱动的mmap接口实现。通过实例展示了两种访问dma-buf物理内存的方式。">
              <link rel="stylesheet" type="text/css" href="https://csdnimg.cn/release/blogv2/dist/pc/css/detail_enter-8834632c0c.min.css">
    <style>
        #content_views pre{
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            -khtml-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none; 
            user-select: none; 
        }
        #content_views pre code{
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            -khtml-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none; 
            user-select: none; 
        }
    </style>
    <script type="application/ld+json">{"@context":"https://ziyuan.baidu.com/contexts/cambrian.jsonld","@id":"https://blog.csdn.net/hexiaolong2009/article/details/102596791","appid":"1638831770136827","pubDate":"2019-11-26T00:12:20","title":"dma-buf 由浅入深（四） &mdash;&mdash; mmap_dmabuf vmap-CSDN博客","upDate":"2019-11-26T00:12:20"}</script>
        <link rel="stylesheet" type="text/css" href="https://csdnimg.cn/release/blogv2/dist/pc/themesSkin/skin-yellow/skin-yellow-28d34ab5fa.min.css">
    <script src="https://g.csdnimg.cn/lib/jquery/1.12.4/jquery.min.js" type="text/javascript"></script>
    <script src="https://g.csdnimg.cn/lib/jquery-migrate/1.4.1/jquery-migrate.js" type="text/javascript"></script>
    <script type="text/javascript">
        var isCorporate = false;
        var username =  "hexiaolong2009";
        var skinImg = "white";

        var blog_address = "https://blog.csdn.net/hexiaolong2009";
        var currentUserName = "";
        var isOwner = false;
        var loginUrl = "http://passport.csdn.net/account/login?from=https://blog.csdn.net/hexiaolong2009/article/details/102596791";
        var blogUrl = "https://blog.csdn.net/";
        var starMapUrl = "https://ai.csdn.net";
        var inscodeHost = "https://inscode.csdn.net";
        var paymentBalanceUrl = "https://csdnimg.cn/release/vip-business-components/vipPaymentBalance.js";
        var appBlogDomain = "https://app-blog.csdn.net";
        var avatar = "https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1";
        var isCJBlog = false;
        var isStarMap = true;
        var articleTitle = "dma-buf 由浅入深（四） —— mmap";
        var articleDesc = "文章浏览阅读2.7w次，点赞24次，收藏54次。本文深入探讨了dma-buf在用户空间的访问方法，包括直接使用fd进行mmap操作和利用dma_buf_mmap简化设备驱动的mmap接口实现。通过实例展示了两种访问dma-buf物理内存的方式。";
        var articleTitles = "dma-buf 由浅入深（四） —— mmap_dmabuf vmap-CSDN博客";
        var nickName = "何小龙";
        var articleDetailUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596791";
        var vipUrlV = "https://mall.csdn.net/vip?vipSource=learningVip";
        if(window.location.host.split('.').length == 3) {
            blog_address = blogUrl + username;
        }
        var skinStatus = "White";
        var blogStaticHost = "https://csdnimg.cn/release/blogv2/"
          var payColumn = false
    </script>
        <meta name="toolbar" content='{"type":"0","fixModel":"1"}'>
    <script src="https://g.csdnimg.cn/??fixed-sidebar/1.1.7/fixed-sidebar.js" type="text/javascript"></script>
    <script src='//g.csdnimg.cn/common/csdn-report/report.js' type='text/javascript'></script>
    <link rel="stylesheet" type="text/css" href="https://csdnimg.cn/public/sandalstrap/1.4/css/sandalstrap.min.css">
    <style>
        .MathJax, .MathJax_Message, .MathJax_Preview{
            display: none
        }
    </style>
    <script src="https://dup.baidustatic.com/js/ds.js"></script>
      <script type="text/javascript">
        (function(c,l,a,r,i,t,y){
            c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        })(window, document, "clarity", "script", "lgtpix6r85");
      </script>
    <script src="/cdn_cgi_bs_bot/static/crypto.min.js"></script>
    <script src="/cdn_cgi_bs_bot/static/bot-score-v1.js"></script>
    <script src="/cdn_cgi_bs_captcha/static/js/waf_captcha_embedded_bs.js"></script>
</head>
  <body class="nodata  " style="">
    <div id="toolbarBox" style="min-height: 48px;"></div>
        <script>
            var toolbarSearchExt = '{\"id\":102596791,\"landingWord\":[\"dmabuf vmap\"],\"queryWord\":\"\",\"tag\":[\"dma-buf\",\"DRM\",\"内存管理\"],\"title\":\"dma-buf 由浅入深（四） &mdash;&mdash; mmap\"}';
        </script>
      <script src="https://g.csdnimg.cn/common/csdn-toolbar/csdn-toolbar.js" type="text/javascript"></script>
    <script>
    (function(){
        var bp = document.createElement('script');
        var curProtocol = window.location.protocol.split(':')[0];
        if (curProtocol === 'https') {
            bp.src = 'https://zz.bdstatic.com/linksubmit/push.js';
        }
        else {
            bp.src = 'http://push.zhanzhang.baidu.com/push.js';
        }
        var s = document.getElementsByTagName("script")[0];
        s.parentNode.insertBefore(bp, s);
    })();
    </script>

    <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/pc/css/blog_code-01256533b5.min.css">
    <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/chart-3456820cac.css" />
    <link rel="stylesheet" href="https://g.csdnimg.cn/lib/swiper/6.0.4/css/swiper.css" />
    <script src="https://g.csdnimg.cn/lib/swiper/6.0.4/js/swiper.js" async></script>
    <script>
      var articleId = 102596791;
        var privateEduData = [];
        var privateData = ["linux","api","参考资料","内存","cpu"];//高亮数组
      var crytojs = "https://csdnimg.cn/release/blogv2/dist/components/js/crytojs-ca5b8bf6ae.min.js";
      var commentscount = 14;
      var commentAuth = 2;
      var curentUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596791";
      var myUrl = "https://my.csdn.net/";
      var isGitCodeBlog = false;
      var vipActivityIcon = "https://i-operation.csdnimg.cn/images/df6c67fa661c48eba86beaeb64350df0.gif";
      var isOpenSourceBlog = false;
      var isVipArticle = false;
        var highlight = ["dmabuf","vmap","由浅入深","mmap","内存管理","buf","dma","drm","(",")","四","-"];//高亮数组
        var isRecommendModule = true;
          var isBaiduPre = true;
          var baiduCount = 2;
          var setBaiduJsCount = 10;
        var viewCountFormat = 27384;
      var share_card_url = "https://app-blog.csdn.net/share?article_id=102596791&username=hexiaolong2009"
      var mallVipUrl = "https://mall.csdn.net/vip?vipSource=article"
      var vipArticleAbStyle = "t_1"
      var vipArticleCpStyle = "t_1"
      var detailheaderAbCommunity = "control"
      var codeAiAbStyle = "exp1"
      var runProjectDeepBlogAB = "isRunProject is false"
      var gitcodeHighlightWordAB = "exp1"
      var deepblogUrl = ""
        deepblogUrl = "https://deepblog.net?utm_source=blog_onekey_run";
      var codeAiAbObjStyle = ""
        codeAiAbObjStyle = "{\"control\":{\"title\":\"AI写代码\",\"destUrl\":\"https://trae.com.cn?utm_source=community&utm_medium=csdn&utm_campaign=daima\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/a5fff6f6c9f0464c9a46b130c972952b.png\"},\"exp1\":{\"title\":\"一键获取完整项目代码\",\"runClose\":true,\"destUrl\":\"https://inscode.net?utm_source=blog_code_block\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/25e1eba3e6bc4df7ba20f2b6011fbe21.png\",\"blogUrl\":true},\"control_run_project\": {\"title\": \"运行项目并下载源码\",\"destUrl\": \"\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/46c457a2cf8b4b9b8f17a2ab71461d4a.png\"},\"control_deepblog\": {\"title\": \"AI生成项目\",\"destUrl\": \"https://inscode.net?utm_source=blog_code_block_fixed\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/9899ea0f099e4e4e8b1fcdb918a27fcd.png\",\"blogUrl\": true},\"exp2\":{\"title\":\"智能体编程\",\"destUrl\":\"https://qoder.com/referral?referral_code=kyKxftaZjisNKOow777DARC0j35axVBq\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/afa15dc565924b96a783f4db37687511.png\"},\"exp3\":{\"title\":\"AI构建项目\",\"destUrl\":\"https://t.csdnimg.cn/9I17\",\"imgUrl\":\"https://i-operation.csdnimg.cn/images/bf18ebff2c3748d59ea0f95954bb4b4a.png\"}}";
      var aiSideSegment = -1;
        aiSideSegment = "16";
      var articleType = 1;
      var baiduKey = "dmabuf vmap";
      var copyPopSwitch = true;
      var needInsertBaidu = true;
      var recommendRegularDomainArr = ["blog.csdn.net/.+/article/details/","download.csdn.net/download/","edu.csdn.net/course/detail/","ask.csdn.net/questions/","bbs.csdn.net/topics/","www.csdn.net/gather_.+/"]
      var codeStyle = "atom-one-dark";
      var baiduSearchType = "baidulandingword";
      var sharData = "{\"hot\":[{\"id\":1,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a5f4260710904e538002a6ab337939b3.png\"},{\"id\":2,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/188b37199a2c4b74b1d9ffc39e0d52de.png\"},{\"id\":3,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/14ded358b631444581edd98a256bc5af.png\"},{\"id\":4,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/1470f23a770444d986ad551b9c33c5be.png\"},{\"id\":5,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c329f5181dc74f6c9bd28c982bb9f91d.png\"},{\"id\":6,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/ccd8a3305e81460f9c505c95b432a65f.png\"},{\"id\":7,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/bc89d8283389440d97fc4d30e30f45e1.png\"},{\"id\":8,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/452d485b4a654f5592390550d2445edf.png\"},{\"id\":9,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/f8b9939db2ed474a8f43a643015fc8b7.png\"},{\"id\":10,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/6de8864187ab4ed3b1db0856369c36ff.png\"},{\"id\":11,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/673cc3470ff74072acba958dc0c46e2d.png\"},{\"id\":12,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/930c119760ac4491804db80f9c6d4e3f.png\"},{\"id\":13,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/15e6befb05a24233bc2b65e96aa8d972.png\"},{\"id\":14,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/2075fd6822184b95a41e214de4daec13.png\"},{\"id\":15,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/859b1552db244eb6891a809263a5c657.png\"},{\"id\":16,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/0be2f920f1f74290a98921974a9613fd.png\"},{\"id\":17,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/2e97e00b43f14afab494ea55ef3f4a6e.png\"},{\"id\":18,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/ff4ab252f46e444686f5135d6ebbfec0.png\"},{\"id\":19,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/ae029bbe99564e79911657912d36524f.png\"},{\"id\":20,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b3ece39963de440388728e9e7b9bf427.png\"},{\"id\":21,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/6f14651a99ba486e926d63b6fa692997.png\"},{\"id\":22,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/83ceddf050084875a341e32dcceca721.png\"},{\"id\":23,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b90368b8fd5d4c6c8c79a707d877cf7c.png\"},{\"id\":24,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/aeffae14ecf14e079b2616528c9a393b.png\"},{\"id\":25,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c5a06b5a13d44d16bed868fc3384897a.png\"},{\"id\":26,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/08b697658b844b318cea3b119e9541ef.png\"},{\"id\":27,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/68ccb0b8d09346ac961d2b5c1a8c77bf.png\"},{\"id\":28,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a2227a247e37418cbe0ea972ba6a859b.png\"},{\"id\":29,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/3a42825fede748f9993e5bb844ad350d.png\"},{\"id\":30,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/8882abc1dd484224b636966ea38555c3.png\"},{\"id\":31,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/4f6a5f636a3e444d83cf8cc06d87a159.png\"},{\"id\":32,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/1953ef79c56b4407b78d7181bdff11c3.png\"},{\"id\":33,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c04a2a4f772948ed85b5b0380ed36287.png\"},{\"id\":34,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/5b4fecd05091405ea04d8c0f53e9f2c7.png\"},{\"id\":35,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b89f576d700344e280d6ceb2a66c2420.png\"},{\"id\":36,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/1c65780e11804bbd9971ebadb3d78bcf.png\"},{\"id\":37,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/d590db2055f345db9706eb68a7ec151a.png\"},{\"id\":38,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/fe602f80700b4f6fb3c4a9e4c135510e.png\"},{\"id\":39,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/39ff2fcd31e04feba301a071976a0ba7.png\"},{\"id\":40,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/f9b61b3d113f436b828631837f89fb39.png\"},{\"id\":41,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/df1aca5f610c4ad48cd16da88c9c8499.png\"},{\"id\":42,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/d7acf73a1e6b41399a77a85040e10961.png\"},{\"id\":43,\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/b7f1b63542524b97962ff649ab4e7e23.png\"}],\"vip\":[{\"id\":1,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101150.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101154.png\"},{\"id\":2,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101204.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101208.png\"},{\"id\":3,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101211.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101215.png\"},{\"id\":4,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101218.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101220.png\"},{\"id\":5,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101223.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220920101226.png\"},{\"id\":6,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100635.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100639.png\"},{\"id\":7,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100642.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100644.png\"},{\"id\":8,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100647.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100649.png\"},{\"id\":9,\"vipUrl\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100652.png\",\"url\":\"https:\\/\\/img-home.csdnimg.cn\\/images\\/20220922100655.png\"},{\"id\":10,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/55de67481fde4b04b97ad78f11fe369a.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/bb2418fb537e4d78b10d8765ccd810c5.png\"},{\"id\":11,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/579c713394584d128104ef1044023954.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/f420d9fbcf5548079d31b5e809b6d6cd.png\"},{\"id\":12,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/75b7f3155ba642f5a4cc16b7baf44122.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a9030f5877be401f8b340b80b0d91e64.png\"},{\"id\":13,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/0903d33cafa54934be3780aa54ae958d.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/2cd8c8929f5a42fca5da2a0aeb456203.png\"},{\"id\":14,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/949fd7c22884439fbfc3c0e9c3b8dee7.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/dafbea9bd9eb4f3b962b48dc41657f89.png\"},{\"id\":15,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/4119cfddd71d4e6a8a27a18dbb74d90e.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/c56310c8b6384d9e85388e4e342ce508.png\"},{\"id\":16,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/121575274da142bcbbbbc2e8243dd411.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/5013993de06542f881018bb9abe2edf7.png\"},{\"id\":17,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/4d97aa6dd4fe4f09a6bef5bdf8a6abcd.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/76f23877b6ad4066ad45ce8e31b4b977.png\"},{\"id\":18,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/fdb619daf21b4c829de63b9ebc78859d.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/a1abe5d27a5441f599adfe662f510243.png\"},{\"id\":19,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/676b7707bb11410f8f56bc0ed2b2345c.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/7ac5b467fbf24e1d8c2de3f3332c4f54.png\"},{\"id\":20,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/0becb8cc227e4723b765bdd69a20fd4a.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/fdec85b26091486b9a89d0b8d45c3749.png\"},{\"id\":21,\"vipUrl\":\"https:\\/\\/img-blog.csdnimg.cn\\/1a6c06235ad44941b38c54cbc25a370c.png\",\"url\":\"https:\\/\\/img-blog.csdnimg.cn\\/410a06cda2d44b0c84578f88275caf70.png\"}],\"map\":{\"hot\":\"热门\",\"vip\":\"VIP\"}}";
        var commentListModule = "true"
      var canRead = true;
      var blogMoveHomeArticle = false;
      var showSearchText = "";
      var articleSource = 1;
      var articleReport = '{"pid": "blog", "spm":"1001.2101"}';
        var baiduSearchChannel = 'pc_relevant'
        var baiduSearchIdentification = '.235^v43^pc_blog_bottom_relevance_base9'
        var distRequestId = '1766382488789_95616'
        var initRewardObject = {
          giver: currentUserName,
          anchor: username,
          articleId: articleId,
          sign: ''
        }
        var isLikeStatus = false;
        var isUnLikeStatus = false;
        var studyLearnWord = "";
        var unUseCount = 0;
        var codeMaxSize = 0;
        var overCost = true;
        var isCurrentUserVip = false;
        var contentViewsHeight = 0;
        var contentViewsCount = 0;
        var contentViewsCountLimit = 5;
        var isShowConcision = false;
        var lastTime = 0
        var postTime = "2019-11-26 00:12:20"
      var isCookieConcision = false
      var isHasDirectoryModel = false
      var isShowSideModel = false
      var isShowDirectoryModel = true
      function getCookieConcision(sName){
        var allCookie = document.cookie.split("; ");
        for (var i=0; i < allCookie.length; i++){
          var aCrumb = allCookie[i].split("=");
          if (sName == aCrumb[0])
            return aCrumb[1];
        }
        return null;
      }
      if (getCookieConcision('blog_details_concision') && getCookieConcision('blog_details_concision') == 0){
        isCookieConcision = true
        isShowSideModel = true
        isShowDirectoryModel = false
      }
    </script>
        <div class="main_father clearfix d-flex justify-content-center " style="height:100%;">
          <div class="container clearfix " id="mainBox">
          <script>
          if (!isCookieConcision) {
            $('.main_father').removeClass('mainfather-concision')
            $('.main_father .container').removeClass('container-concision')
          } else {
            $('#mainBox').css('margin-right', '0')
          }
          </script>
          <main>
<script type="text/javascript">
    var resourceId =  "";
    function getQueryString(name) {   
      var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)"); //构造一个含有目标参数的正则表达式对象  
      var r = window.location.search.substr(1).match(reg);  //匹配目标参数
      if( r != null ) return decodeURIComponent( r[2] ); return '';   
    }
    function stripscript(s){ 
      var pattern = new RegExp("[`~!@#$^&*()=|{}':;',\\[\\].<>/?~！@#￥……&*（）——|{}【】‘；：”“'。，、？%]") 
      var rs = ""; 
      for (var i = 0; i < s.length; i++) { 
        rs = rs+s.substr(i, 1).replace(pattern, ''); 
      } 
      return rs;
    }
    var blogHotWords = stripscript(getQueryString('utm_term')).length > 1 ? stripscript(getQueryString('utm_term')) : ''
</script>
<div class="blog-content-box">
  <div class="article-header-box" id="article-header-box">
    <div class="article-header">
      <div class="article-title-box">
        <h1 class="title-article" id="articleContentId">dma-buf 由浅入深（四） —— mmap</h1>
      </div>
      <div class="article-info-box">
              <div class="up-time">最新推荐文章于&nbsp;2025-11-01 13:39:13&nbsp;发布</div>
          <div class="article-bar-top">
              <div class="bar-content active">
              <span class="article-type-text original">原创</span>
                    <span class="time blog-postTime" data-time="2019-11-26 00:12:20">最新推荐文章于&nbsp;2025-11-01 13:39:13&nbsp;发布</span>
                <span class="border-dian">·</span>
                <span class="read-count">2.7w 阅读</span>
                <div class="read-count-box is-like like-ab-new" data-type="top">
                  <span class="border-dian">·</span>
                  <img class="article-read-img article-heard-img active" style="display:none" id="is-like-imgactive-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png" alt="">
                  <img class="article-read-img article-heard-img" style="display:block" id="is-like-img-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png" alt="">
                  <span class="read-count" id="blog-digg-num" style="color:;">
                      24
                  </span>
                </div>
                <span class="border-dian">·</span>
                <a id="blog_detail_zk_collection" class="un-collection" data-report-click='{"mod":"popu_823","spm":"1001.2101.3001.4232","ab":"new"}'>
                  <img class="article-collect-img article-heard-img un-collect-status isdefault" style="display:inline-block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png" alt="">
                  <img class="article-collect-img article-heard-img collect-status isactive" style="display:none" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png" alt="">
                  <span class="get-collection">
                      54
                  </span>
                </a>

                  <span class="border-dian">·</span>
                  <div class="href-article-edit-new">
                    <span class="href-article-edit-click">CC 4.0 BY-SA版权</span>
                    <div class="slide-content-box-new">
                                版权声明：本文为博主原创文章，遵循<a href="http://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener"> CC 4.0 BY-SA </a>版权协议，转载请附上原文出处链接和本声明。
                    </div>
                  </div>
              </div>
              <div class="operating active">
              </div>
          </div>
          <div class="blog-tags-box">
             
              <div class="tags-box artic-tag-box">
                     <div class="article-tag">
                       <span class="label">文章标签：</span>
                      <p>
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"dma-buf","ab":"new","extra":"{\"searchword\":\"dma-buf\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"dma-buf","ab":"new","extra":"{\"searchword\":\"dma-buf\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=dma-buf&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#dma-buf</a>
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"DRM","ab":"new","extra":"{\"searchword\":\"DRM\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"DRM","ab":"new","extra":"{\"searchword\":\"DRM\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=DRM&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#DRM</a>
                          <a rel="nofollow" data-report-query="spm=1001.2101.3001.4223" data-report-click='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"内存管理","ab":"new","extra":"{\"searchword\":\"内存管理\"}"}' data-report-view='{"mod":"popu_626","spm":"1001.2101.3001.4223","strategy":"内存管理","ab":"new","extra":"{\"searchword\":\"内存管理\"}"}' class="tag-link-new" href="https://so.csdn.net/so/search/s.do?q=%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86&amp;t=all&amp;o=vip&amp;s=&amp;l=&amp;f=&amp;viparticle=&amp;from_tracking_code=tag_word&amp;from_code=app_blog_art" target="_blank" rel="noopener">#内存管理</a>
                      </p>
                     </div>
                  <p class="community-name" id="community-name"></p>
              </div>
          </div>
       
          
       
      </div>
    </div>
  </div>
    <div id="blogHuaweiyunAdvert" class=""></div>
        <div id="blogColumnPayAdvert" class="">
              <div class="column-group">
                <div class="column-group-item column-group0 ">
                    <div class="item-l">
                        <a class="item-target" href="https://blog.csdn.net/hexiaolong2009/category_9281458.html" target="_blank" title="DRM (Direct Rendering Manager)"
                        data-report-view='{"spm":"1001.2101.3001.6332"}'
                        data-report-click='{"spm":"1001.2101.3001.6332"}'>
                            <img class="item-target" src="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_224,w_224" alt="">
                            <span class="title item-target">
                                <span>
                                <span class="tit">DRM (Direct Rendering Manager)</span>
                                    <span class="dec more">同时被 2 个专栏收录<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/newArrowDown1White.png" alt=""></span>
                                </span>
                            </span>
                        </a>
                    </div>
                    <div class="item-m">
                        <span>29 篇文章</span>
                    </div>
                    <div class="item-r">
                            <a class="item-target article-column-bt articleColumnFreeBt" data-id="9281458">订阅专栏</a>
                    </div>
                </div>
                <div class="column-group-item column-group1 ">
                    <div class="item-l">
                        <a class="item-target" href="https://blog.csdn.net/hexiaolong2009/category_10838100.html" target="_blank" title="DMA-BUF"
                        data-report-view='{"spm":"1001.2101.3001.6332"}'
                        data-report-click='{"spm":"1001.2101.3001.6332"}'>
                            <img class="item-target" src="https://i-blog.csdnimg.cn/columns/default/20201014180756923.png?x-oss-process=image/resize,m_fixed,h_224,w_224" alt="">
                            <span class="title item-target">
                                <span>
                                <span class="tit">DMA-BUF</span>
                                </span>
                            </span>
                        </a>
                    </div>
                    <div class="item-m">
                        <span>10 篇文章</span>
                    </div>
                    <div class="item-r">
                            <a class="item-target article-column-bt articleColumnFreeBt" data-id="10838100">订阅专栏</a>
                    </div>
                </div>
              </div>

        </div>
      <div class="ai-abstract-box">
         <div class="ai-abstract">
          <div class="abstract-content">
            <img class="lock-img" src="https://i-operation.csdnimg.cn/images/a7311a21245d4888a669ca3155f1f4e5.png" alt="">本文深入探讨了dma-buf在用户空间的访问方法，包括直接使用fd进行mmap操作和利用dma_buf_mmap简化设备驱动的mmap接口实现。通过实例展示了两种访问dma-buf物理内存的方式。
          </div>
        </div>
      </div>
      <div class="starmap-box box1" data-spm='3001.11251' data-id='gpu_img_ace_step' data-utm-source='top' data-report-view='{"spm":"3001.11251","extra":{"openMirrorId":"gpu_img_ace_step"}}' data-report-click='{"spm":"3001.11251","extra":{"openMirrorId":"gpu_img_ace_step"}}'>
        部署运行你感兴趣的模型镜像<button class="btn-go">一键部署</button>
      </div>
    <article class="baidu_pl">
        <div id="article_content" class="article_content clearfix">
        <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/kdoc_html_views-1a98987dfd.css">
        <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/ck_htmledit_views-10bf609291.css">
                <div id="content_views" class="markdown_views prism-atom-one-dark">
                    <svg xmlns="http://www.w3.org/2000/svg" style="display: none;">
                        <path stroke-linecap="round" d="M5,0 0,2.5 5,5z" id="raphael-marker-block" style="-webkit-tap-highlight-color: rgba(0, 0, 0, 0);"></path>
                    </svg>
                    <p><a href="https://blog.csdn.net/hexiaolong2009/article/details/102596744">dma-buf 由浅入深&#xff08;一&#xff09; —— 最简单的 dma-buf 驱动程序</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596761">dma-buf 由浅入深&#xff08;二&#xff09; —— kmap / vmap</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772">dma-buf 由浅入深&#xff08;三&#xff09; —— map attachment</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791">dma-buf 由浅入深&#xff08;四&#xff09; —— mmap</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802">dma-buf 由浅入深&#xff08;五&#xff09; —— File</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596825">dma-buf 由浅入深&#xff08;六&#xff09; —— begin / end cpu_access</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596845">dma-buf 由浅入深&#xff08;七&#xff09; —— alloc page 版本</a><br /> <a href="https://blog.csdn.net/hexiaolong2009/article/details/103795381">dma-buf 由浅入深&#xff08;八&#xff09; —— ION 简化版</a></p> 
<hr /> 
<h3><a id="_11"></a>前言</h3> 
<p>前面的两篇文章<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596761">《dma-buf 由浅入深&#xff08;二&#xff09; —— kmap/vmap》</a>和<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772">《dma-buf 由浅入深&#xff08;三&#xff09; —— map attachment》</a>都是在 kernel space 对 dma-buf 进行访问的&#xff0c;本篇我们将一起来学习&#xff0c;如何在 user space 访问 dma-buf。当然&#xff0c;user space 访问 dma-buf 也属于 CPU Access 的一种。</p> 
<h3><a id="mmap_14"></a>mmap</h3> 
<p>为了方便应用程序能直接在用户空间读写 dma-buf 的内存&#xff0c;<em>dma_buf_ops</em> 为我们提供了一个 <em>mmap</em> 回调接口&#xff0c;可以把 dma-buf 的物理内存直接映射到用户空间&#xff0c;这样应用程序就可以像访问普通文件那样访问 dma-buf 的物理内存了。</p> 
<blockquote> 
 <p><a href="https://github.com/torvalds/linux/commit/4c78513e457f72d5554a0f6e2eabfad7b98e4f19">dma-buf: mmap support</a></p> 
</blockquote> 
<p>在 Linux 设备驱动中&#xff0c;大多数驱动的 <em>mmap</em> 操作接口都是通过调用 <code>remap_pfn_range()</code> 函数来实现的&#xff0c;dma-buf 也不例外。对于此函数不了解的同学&#xff0c;推荐阅读 <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791#t6"><em>参考资料</em></a> 中彭东林的博客&#xff0c;写的非常好&#xff01;</p> 
<p>除了 <em>dma_buf_ops</em> 提供的 <em>mmap</em> 回调接口外&#xff0c;dma-buf 还为我们提供了 <code>dma_buf_mmap()</code> 内核 API&#xff0c;使得我们可以在其他设备驱动中就地取材&#xff0c;直接引用 dma-buf 的 <em>mmap</em> 实现&#xff0c;以此来间接的实现设备驱动的 <em>mmap</em> 文件操作接口。</p> 
<h3><a id="_24"></a>示例</h3> 
<p><img src="https://i-blog.csdnimg.cn/blog_migrate/955754811cba0b2b6a87fbc770b468a7.png#pic_center" alt="在这里插入图片描述" width="515" height="296" /></p> 
<p>接下来&#xff0c;我们将通过两个示例来演示如何在 Userspace 访问 dma-buf 的物理内存。</p> 
<ul><li>示例一&#xff1a;直接使用 dma-buf 的 fd 做 <em>mmap()</em> 操作</li><li>示例二&#xff1a;使用 exporter 的 fd 做 <em>mmap()</em> 操作</li></ul> 
<h4><a id="_31"></a>示例一</h4> 
<p>本示例主要演示如何在驱动层实现 dma-buf 的 <em>mmap</em> 回调接口&#xff0c;以及如何在用户空间直接使用 dma-buf 的 fd 进行 <em>mmap()</em> 操作。</p> 
<h5><a id="exporter__34"></a>exporter 驱动</h5> 
<p>首先&#xff0c;我们仍然基于第一篇的 exporter-dummy.c 驱动来实现 <em>mmap</em> 回调接口&#xff1a;</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/04/exporter-fd.c">exporter-fd.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/miscdevice.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/slab.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/uaccess.h&gt;</span></span>

<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf_exported<span class="token punctuation">;</span>
<span class="token function">EXPORT_SYMBOL</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_mmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span> <span class="token keyword">struct</span> vm_area_struct <span class="token operator">*</span>vma<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">void</span> <span class="token operator">*</span>vaddr <span class="token operator">&#61;</span> dmabuf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token function">remap_pfn_range</span><span class="token punctuation">(</span>vma<span class="token punctuation">,</span> vma<span class="token operator">-&gt;</span>vm_start<span class="token punctuation">,</span> <span class="token function">virt_to_pfn</span><span class="token punctuation">(</span>vaddr<span class="token punctuation">)</span><span class="token punctuation">,</span>
				PAGE_SIZE<span class="token punctuation">,</span> vma<span class="token operator">-&gt;</span>vm_page_prot<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">static</span> <span class="token keyword">const</span> <span class="token keyword">struct</span> dma_buf_ops exp_dmabuf_ops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
	<span class="token punctuation">.</span>mmap <span class="token operator">&#61;</span> exporter_mmap<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> dma_buf <span class="token operator">*</span><span class="token function">exporter_alloc_page</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">DEFINE_DMA_BUF_EXPORT_INFO</span><span class="token punctuation">(</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">;</span>
	<span class="token keyword">void</span> <span class="token operator">*</span>vaddr<span class="token punctuation">;</span>

	vaddr <span class="token operator">&#61;</span> <span class="token function">kzalloc</span><span class="token punctuation">(</span>PAGE_SIZE<span class="token punctuation">,</span> GFP_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>

	exp_info<span class="token punctuation">.</span>ops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>exp_dmabuf_ops<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>size <span class="token operator">&#61;</span> PAGE_SIZE<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> O_CLOEXEC<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>priv <span class="token operator">&#61;</span> vaddr<span class="token punctuation">;</span>

	dmabuf <span class="token operator">&#61;</span> <span class="token function">dma_buf_export</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">sprintf</span><span class="token punctuation">(</span>vaddr<span class="token punctuation">,</span> <span class="token string">&#34;hello world!&#34;</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> dmabuf<span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">long</span> <span class="token function">exporter_ioctl</span><span class="token punctuation">(</span><span class="token keyword">struct</span> file <span class="token operator">*</span>filp<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">int</span> cmd<span class="token punctuation">,</span> <span class="token keyword">unsigned</span> <span class="token keyword">long</span> arg<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd <span class="token operator">&#61;</span> <span class="token function">dma_buf_fd</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">,</span> O_CLOEXEC<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">copy_to_user</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token keyword">int</span> __user <span class="token operator">*</span><span class="token punctuation">)</span>arg<span class="token punctuation">,</span> <span class="token operator">&amp;</span>fd<span class="token punctuation">,</span> <span class="token keyword">sizeof</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> file_operations exporter_fops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>owner		<span class="token operator">&#61;</span> THIS_MODULE<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>unlocked_ioctl	<span class="token operator">&#61;</span> exporter_ioctl<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> miscdevice mdev <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>minor <span class="token operator">&#61;</span> MISC_DYNAMIC_MINOR<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>name <span class="token operator">&#61;</span> <span class="token string">&#34;exporter&#34;</span><span class="token punctuation">,</span>
	<span class="token punctuation">.</span>fops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>exporter_fops<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">int</span> __init <span class="token function">exporter_init</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	dmabuf_exported <span class="token operator">&#61;</span> <span class="token function">exporter_alloc_page</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">return</span> <span class="token function">misc_register</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> __exit <span class="token function">exporter_exit</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">misc_deregister</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token function">module_init</span><span class="token punctuation">(</span>exporter_init<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token function">module_exit</span><span class="token punctuation">(</span>exporter_exit<span class="token punctuation">)</span><span class="token punctuation">;</span>

</code></pre> 
<p>从上面的示例可以看到&#xff0c;除了要实现 dma-buf 的 <em>mmap</em> 回调接口外&#xff0c;我们还引入了 misc driver&#xff0c;目的是想通过 misc driver 的 <em>ioctl</em> 接口将 dma-buf 的 fd 传递给上层应用程序&#xff0c;这样才能实现应用程序<strong>直接</strong>使用这个 dma-buf fd 做 <em>mmap()</em> 操作。</p> 
<blockquote> 
 <p>为什么非要通过 ioctl 的方式来传递 fd &#xff1f;这个问题我会在下一篇<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802">《dma-buf 由浅入深&#xff08;五&#xff09;—— File》</a>中详细讨论。</p> 
</blockquote> 
<p>在 <em>ioctl</em> 接口中&#xff0c;我们使用到了 <code>dma_buf_fd()</code> 函数&#xff0c;该函数用于创建一个新的 fd&#xff0c;并与该 dma-buf 的文件相绑定。关于该函数&#xff0c;我也会在下一篇中做详细介绍。</p> 
<h5><a id="userspace__124"></a>userspace 程序</h5> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/04/dmabuf-test/mmap_dmabuf.c">mmap_dmabuf.c</a></p> 
<pre><code class="prism language-c"><span class="token keyword">int</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token keyword">int</span> argc<span class="token punctuation">,</span> <span class="token keyword">char</span> <span class="token operator">*</span>argv<span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd<span class="token punctuation">;</span>
	<span class="token keyword">int</span> dmabuf_fd <span class="token operator">&#61;</span> <span class="token number">0</span><span class="token punctuation">;</span>

	fd <span class="token operator">&#61;</span> <span class="token function">open</span><span class="token punctuation">(</span><span class="token string">&#34;/dev/exporter&#34;</span><span class="token punctuation">,</span> O_RDONLY<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">ioctl</span><span class="token punctuation">(</span>fd<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token operator">&amp;</span>dmabuf_fd<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">close</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">char</span> <span class="token operator">*</span>str <span class="token operator">&#61;</span> <span class="token function">mmap</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> <span class="token number">4096</span><span class="token punctuation">,</span> PROT_READ<span class="token punctuation">,</span> MAP_SHARED<span class="token punctuation">,</span> dmabuf_fd<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">printf</span><span class="token punctuation">(</span><span class="token string">&#34;read from dmabuf mmap: %s\n&#34;</span><span class="token punctuation">,</span> str<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
</code></pre> 
<p>可以看到 userspace 的代码非常简单&#xff0c;首先通过 exporter 驱动的 <em>ioctl()</em> 获取到 dma-buf 的 fd&#xff0c;然后直接使用该 fd 做 <em>mmap()</em> 映射&#xff0c;最后使用 <em>printf()</em> 来输出映射后的 buffer 内容。</p> 
<h5><a id="_145"></a>运行结果</h5> 
<p>在 my-qemu 仿真环境中执行如下命令&#xff1a;</p> 
<pre><code class="prism language-handlebars"><span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/exporter-fd.ko</span>
<span class="token punctuation">#</span> <span class="token punctuation">.</span><span class="token punctuation">/</span><span class="token variable">mmap_dmabuf</span>
</code></pre> 
<p>将看到如下打印结果&#xff1a;</p> 
<pre><code>read from dmabuf mmap: hello world!
</code></pre> 
<p>可以看到&#xff0c;userspace 程序通过 <em>mmap()</em> 接口成功的访问到 dma-buf 的物理内存。</p> 
<blockquote> 
 <p>关于应用程序直接使用 dma-buf fd 做 <em>mmap()</em> 操作的案例&#xff0c;Google ADF 的 simple_buffer_alloc 可谓在这一点上发挥的淋漓尽致&#xff01;详细参考代码如下&#xff1a;</p> 
 <ul><li><a href="http://androidxref.com/kernel_3.18/xref/drivers/video/adf/adf_fops.c#379" rel="nofollow">kernel-3.18/drivers/video/adf/adf_fops.c</a></li><li><a href="http://androidxref.com/9.0.0_r3/xref/bootable/recovery/minui/graphics_adf.cpp#49" rel="nofollow">android9_r3/bootable/recovery/minui/graphics_adf.cpp</a></li></ul> 
 <p><br /> 备注&#xff1a;上层 minui 获取到的 surf-&gt;fd 其实就是 dma-buf 的 fd。Recovery 模式下应用程序绘图本质上就是 CPU 通过 mmap() 来操作 dma-buf 的物理内存。</p> 
</blockquote> 
<br /> 
<h4><a id="_167"></a>示例二</h4> 
<p>本示例主要演示如何使用 <code>dma_buf_mmap()</code> 内核 API&#xff0c;以此来简化设备驱动的 <em>mmap</em> 文件操作接口的实现。</p> 
<h5><a id="exporter__170"></a>exporter 驱动</h5> 
<p>我们基于示例一中的 exporter-fd.c 文件&#xff0c;删除 <em>exporter_ioctl()</em> 函数&#xff0c;新增 <em>exporter_misc_mmap()</em> 函数&#xff0c; 具体修改如下&#xff1a;</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/05/exporter-mmap.c">exporter-mmap.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/miscdevice.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/slab.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/uaccess.h&gt;</span></span>

<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf_exported<span class="token punctuation">;</span>
<span class="token function">EXPORT_SYMBOL</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_mmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span> <span class="token keyword">struct</span> vm_area_struct <span class="token operator">*</span>vma<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">void</span> <span class="token operator">*</span>vaddr <span class="token operator">&#61;</span> dmabuf<span class="token operator">-&gt;</span>priv<span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token function">remap_pfn_range</span><span class="token punctuation">(</span>vma<span class="token punctuation">,</span> vma<span class="token operator">-&gt;</span>vm_start<span class="token punctuation">,</span> <span class="token function">virt_to_pfn</span><span class="token punctuation">(</span>vaddr<span class="token punctuation">)</span><span class="token punctuation">,</span>
				PAGE_SIZE<span class="token punctuation">,</span> vma<span class="token operator">-&gt;</span>vm_page_prot<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
<span class="token keyword">static</span> <span class="token keyword">const</span> <span class="token keyword">struct</span> dma_buf_ops exp_dmabuf_ops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
	<span class="token punctuation">.</span>mmap <span class="token operator">&#61;</span> exporter_mmap<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> dma_buf <span class="token operator">*</span><span class="token function">exporter_alloc_page</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">DEFINE_DMA_BUF_EXPORT_INFO</span><span class="token punctuation">(</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">;</span>
	<span class="token keyword">void</span> <span class="token operator">*</span>vaddr<span class="token punctuation">;</span>

	vaddr <span class="token operator">&#61;</span> <span class="token function">kzalloc</span><span class="token punctuation">(</span>PAGE_SIZE<span class="token punctuation">,</span> GFP_KERNEL<span class="token punctuation">)</span><span class="token punctuation">;</span>

	exp_info<span class="token punctuation">.</span>ops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>exp_dmabuf_ops<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>size <span class="token operator">&#61;</span> PAGE_SIZE<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> O_CLOEXEC<span class="token punctuation">;</span>
	exp_info<span class="token punctuation">.</span>priv <span class="token operator">&#61;</span> vaddr<span class="token punctuation">;</span>

	dmabuf <span class="token operator">&#61;</span> <span class="token function">dma_buf_export</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>exp_info<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">sprintf</span><span class="token punctuation">(</span>vaddr<span class="token punctuation">,</span> <span class="token string">&#34;hello world!&#34;</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> dmabuf<span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_misc_mmap</span><span class="token punctuation">(</span><span class="token keyword">struct</span> file <span class="token operator">*</span>file<span class="token punctuation">,</span> <span class="token keyword">struct</span> vm_area_struct <span class="token operator">*</span>vma<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">return</span> <span class="token function">dma_buf_mmap</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">,</span> vma<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">struct</span> file_operations exporter_fops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>owner	<span class="token operator">&#61;</span> THIS_MODULE<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>mmap	<span class="token operator">&#61;</span> exporter_misc_mmap<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">struct</span> miscdevice mdev <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span>minor <span class="token operator">&#61;</span> MISC_DYNAMIC_MINOR<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>name <span class="token operator">&#61;</span> <span class="token string">&#34;exporter&#34;</span><span class="token punctuation">,</span>
	<span class="token punctuation">.</span>fops <span class="token operator">&#61;</span> <span class="token operator">&amp;</span>exporter_fops<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
 
<span class="token keyword">static</span> <span class="token keyword">int</span> __init <span class="token function">exporter_init</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	dmabuf_exported <span class="token operator">&#61;</span> <span class="token function">exporter_alloc_page</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token keyword">return</span> <span class="token function">misc_register</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">void</span> __exit <span class="token function">exporter_exit</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token function">misc_deregister</span><span class="token punctuation">(</span><span class="token operator">&amp;</span>mdev<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token function">module_init</span><span class="token punctuation">(</span>exporter_init<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token function">module_exit</span><span class="token punctuation">(</span>exporter_exit<span class="token punctuation">)</span><span class="token punctuation">;</span>
</code></pre> 
<p>与示例一的驱动相比&#xff0c;示例二的驱动不再需要把 dma-buf 的 fd 通过 ioctl 传给上层&#xff0c;而是直接将 dma-buf 的 mmap 回调接口嫁接到 misc driver 的 mmap 文件操作接口上。这样上层在对 misc device 进行 mmap() 操作时&#xff0c;实际映射的是 dma-buf 的物理内存。</p> 
<h5><a id="userspace__251"></a>userspace 程序</h5> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/05/dmabuf-test/mmap_exporter.c">mmap_exporter.c</a></p> 
<pre><code class="prism language-c"><span class="token keyword">int</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token keyword">int</span> argc<span class="token punctuation">,</span> <span class="token keyword">char</span> <span class="token operator">*</span>argv<span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd<span class="token punctuation">;</span>

	fd <span class="token operator">&#61;</span> <span class="token function">open</span><span class="token punctuation">(</span><span class="token string">&#34;/dev/exporter&#34;</span><span class="token punctuation">,</span> O_RDONLY<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">char</span> <span class="token operator">*</span>str <span class="token operator">&#61;</span> <span class="token function">mmap</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> <span class="token number">4096</span><span class="token punctuation">,</span> PROT_READ<span class="token punctuation">,</span> MAP_SHARED<span class="token punctuation">,</span> fd<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">printf</span><span class="token punctuation">(</span><span class="token string">&#34;read from /dev/exporter mmap: %s\n&#34;</span><span class="token punctuation">,</span> str<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">close</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
</code></pre> 
<p>与示例一的 userspace 程序相比&#xff0c;示例二不再通过 <em>ioctl()</em> 方式获取 dma-buf 的 fd&#xff0c;而是直接使用 exporter misc device 的 fd 进行 <em>mmap()</em> 操作&#xff0c;此时执行的则是 misc driver 的 <em>mmap</em> 文件操作接口。当然最终输出的结果都是一样的。</p> 
<h5><a id="_271"></a>运行结果</h5> 
<p>在 my-qemu 仿真环境中执行如下命令&#xff1a;</p> 
<pre><code class="prism language-handlebars"><span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/exporter-mmap.ko</span>
<span class="token punctuation">#</span> <span class="token punctuation">.</span><span class="token punctuation">/</span><span class="token variable">mmap_exporter</span>
</code></pre> 
<p>将看到如下打印结果&#xff1a;</p> 
<pre><code>read from /dev/exporter mmap: hello world!
</code></pre> 
<br /> 
<h3><a id="_284"></a>开发环境</h3> 
<table><thead><tr><th align="left"></th><th align="left"></th></tr></thead><tbody><tr><td align="left">内核源码</td><td align="left"><a href="https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x/linux-4.14.143.tar.xz" rel="nofollow">4.14.143</a></td></tr><tr><td align="left">示例源码</td><td align="left"><a href="https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/04">hexiaolong2008-GitHub/sample-code/dma-buf/04</a> <br /> <a href="https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/05">hexiaolong2008-GitHub/sample-code/dma-buf/05</a></td></tr><tr><td align="left">开发平台</td><td align="left">Ubuntu14.04/16.04</td></tr><tr><td align="left">运行平台</td><td align="left"><a href="https://github.com/hexiaolong2008/my-qemu">my-qemu 仿真环境</a></td></tr></tbody></table>
<h3><a id="_292"></a>参考资料</h3> 
<ol><li><a href="https://www.cnblogs.com/huxiao-tee/p/4660352.html" rel="nofollow">认真分析mmap&#xff1a;是什么 为什么 怎么用</a></li><li><a href="https://www.cnblogs.com/pengdonglin137/p/8149859.html" rel="nofollow">内存映射函数remap_pfn_range学习——示例分析&#xff08;1&#xff09;</a></li></ol> 
<br /> 
<br /> 
<br /> 
<p>上一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772">《dma-buf 由浅入深&#xff08;三&#xff09;—— map attachment》</a><br /> 下一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802">《dma-buf 由浅入深&#xff08;五&#xff09;—— File》</a><br /> 文章汇总&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/83720940">《DRM&#xff08;Direct Rendering Manager&#xff09;学习简介》</a></p>
                </div>
                <link href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/markdown_views-375c595788.css" rel="stylesheet">
                <link href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/style-e504d6a974.css" rel="stylesheet">
        </div>
    </article>

      <div class="starmap-box box2" data-spm='3001.11252' data-id='gpu_img_ace_step' data-utm-source='bottom' data-report-view='{"spm":"3001.11252","extra":{"openMirrorId":"gpu_img_ace_step"}}' data-report-click='{"spm":"3001.11252","extra":{"openMirrorId":"gpu_img_ace_step"}}'>
        <p class="starmap-title">您可能感兴趣的与本文相关的镜像</p>
        <div class="starmap-content">
          <div class="starmap-info-box">
            <div class="img-box">
              <img src="https://csdn-665-inscode.s3.cn-north-1.jdcloud-oss.com/image/cover/gpu_img_ace_step.png/middle" alt="ACE-Step">
            </div>
            <div class="info-box">
              <div class="title-box">
                <p class="title">ACE-Step</p>
                <div class="tag-box">
                    <div class="tag-item">音乐合成</div>
                    <div class="tag-item">ACE-Step</div>
                </div>
              </div>
              <p class="desc" title="ACE-Step是由中国团队阶跃星辰（StepFun）与ACE Studio联手打造的开源音乐生成模型。 它拥有3.5B参数量，支持快速高质量生成、强可控性和易于拓展的特点。 最厉害的是，它可以生成多种语言的歌曲，包括但不限于中文、英文、日文等19种语言">ACE-Step是由中国团队阶跃星辰（StepFun）与ACE Studio联手打造的开源音乐生成模型。 它拥有3.5B参数量，支持快速高质量生成、强可控性和易于拓展的特点。 最厉害的是，它可以生成多种语言的歌曲，包括但不限于中文、英文、日文等19种语言</p>
            </div>
          </div>
          <div class="starmap-operate-box">
            <button class="starmap-operate-btn">一键部署运行</button>
          </div>
        </div>
      </div>
  <script>
    $(function() {
      setTimeout(function () {
        var mathcodeList = document.querySelectorAll('.htmledit_views img.mathcode');
        if (mathcodeList.length > 0) {
          for (let i = 0; i < mathcodeList.length; i++) {
            if (mathcodeList[i].complete) {
              if (mathcodeList[i].naturalWidth === 0 || mathcodeList[i].naturalHeight === 0) {
                var alt = mathcodeList[i].alt;
                alt = '\\(' + alt + '\\)';
                var curSpan = $('<span class="img-codecogs"></span>');
                curSpan.text(alt);
                $(mathcodeList[i]).before(curSpan);
                $(mathcodeList[i]).remove();
              }
            } else {
              mathcodeList[i].onerror = function() {
                var alt = mathcodeList[i].alt;
                alt = '\\(' + alt + '\\)';
                var curSpan = $('<span class="img-codecogs"></span>');
                curSpan.text(alt);
                $(mathcodeList[i]).before(curSpan);
                $(mathcodeList[i]).remove();
              };
            }
          }
          MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
        }
      }, 500)
    });
  </script>
</div>
<div class="directory-boxshadow-dialog" style="display:none;">
  <div class="directory-boxshadow-dialog-box">
  </div>
   <div class="vip-limited-time-offer-box-new" id="vip-limited-time-offer-box-new">
      <img class="limited-img limited-img-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png">
      <div class="vip-limited-time-top">
        确定要放弃本次机会？
      </div>
      <span class="vip-limited-time-text">福利倒计时</span>
      <div class="limited-time-box-new">
        <span class="time-hour"></span>
        <i>:</i>
        <span class="time-minite"></span>
        <i>:</i>
        <span class="time-second"></span>
      </div>
      <div class="limited-time-vip-box">
        <p>
          <img class="coupon-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png">
          <span class="def">立减 ¥</span>
          <span class="active limited-num"></span>
        </p>
        <span class="">普通VIP年卡可用</span>
      </div>
      <a class="limited-time-btn-new" href="https://mall.csdn.net/vip" data-report-click='{"spm":"1001.2101.3001.9621"}' data-report-query='spm=1001.2101.3001.9621'>立即使用</a>
  </div>
</div>
    <div class="more-toolbox-new more-toolbar" id="toolBarBox">
      <div class="left-toolbox">
        <div class="toolbox-left">
            <div class="profile-box">
              <a class="profile-href" target="_blank" href="https://blog.csdn.net/hexiaolong2009"><img class="profile-img" src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1">
                <span class="profile-name">
                  何小龙
                </span>
              </a>
            </div>
            <div class="profile-attend">
                <a class="tool-attend tool-bt-button tool-bt-attend" href="javascript:;" data-report-view='{"mod":"1592215036_002","spm":"1001.2101.3001.4232","extend1":"关注"}'>关注</a>
              <a class="tool-item-follow active-animation" style="display:none;">关注</a>
            </div>
        </div>
        <div class="toolbox-middle">
          <ul class="toolbox-list">
            <li class="tool-item tool-item-size tool-active is-like" id="is-like" data-type="bottom">
              <a class="tool-item-href">
                <img style="display:none;" id="is-like-imgactive-animation-like" class="animation-dom active-animation" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png" alt="">
                <img class="isactive" style="display:none" id="is-like-imgactive" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png" alt="">
                <img class="isdefault" style="display:block" id="is-like-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png" alt="">
                <span id="spanCount" class="count ">
                    24
                </span>
              </a>
              <div class="tool-hover-tip"><span class="text space">点赞</span></div>
            </li>
            <li class="tool-item tool-item-size tool-active is-unlike" id="is-unlike">
              <a class="tool-item-href">
                <img class="isactive" style="margin-right:0px;display:none" id="is-unlike-imgactive" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png" alt="">
                <img class="isdefault" style="margin-right:0px;display:block" id="is-unlike-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png" alt="">
                <span id="unlikeCount" class="count "></span>
              </a>
              <div class="tool-hover-tip"><span class="text space">踩</span></div>
            </li>
            <li class="tool-item tool-item-size tool-active is-collection ">
              <a class="tool-item-href" href="javascript:;" data-report-click='{"mod":"popu_824","spm":"1001.2101.3001.4130","ab":"new"}'>
                <img style="display:none" id="is-collection-img-collection" class="animation-dom active-animation" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png" alt="">
                <img class="isdefault" id="is-collection-img" style="display:block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png" alt="">
                <img class="isactive" id="is-collection-imgactive" style="display:none" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png" alt="">
                <span class="count get-collection " data-num="54" id="get-collection">
                    54
                </span>
              </a>
              <div class="tool-hover-tip collect">
                <div class="collect-operate-box">
                  <span class="collect-text" id="is-collection">
                    收藏
                  </span>
                </div>
              </div>
              <div class="tool-active-list">
                <div class="text">
                  觉得还不错?
                  <span class="collect-text" id="tool-active-list-collection">
                    一键收藏
                  </span>
                 <img id="tool-active-list-close" src="https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png" alt="">
                </div>
              </div>
            </li>
            <li class="tool-item tool-item-size tool-active tool-item-comment">
              <div class="guide-rr-first">
                <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/guideRedReward01.png" alt="">
                <button class="btn-guide-known">知道了</button>
              </div>
                <a class="tool-item-href go-side-comment" data-report-click='{"spm":"1001.2101.3001.7009"}'>
                <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png" alt="">
                <span class="count">
                    14
                </span>
              </a>
              <div class="tool-hover-tip"><span class="text space">评论</span></div>
            </li>
            <li class="tool-item tool-item-size tool-active tool-QRcode" data-type="article" id="tool-share">
              <a class="tool-item-href" href="javascript:;" data-report-view='{"spm":"3001.4129","extra":{"type":"blogdetail"}}'>
                <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/share.png" alt="">
                <span class="count">分享</span>
              </a>
                <div class="QRcode" id="tool-QRcode">
                <div class="share-bg-box">
                  <div class="share-content">
                    <a id="copyPosterUrl" data-type="link" class="btn-share">复制链接</a>
                  </div>
                  <div class="share-content">
                    <a class="btn-share" data-type="qq">分享到 QQ</a>
                  </div>
                  <div class="share-content">
                    <a class="btn-share" data-type="weibo">分享到新浪微博</a>
                  </div>
                  <div class="share-code">
                    <div class="share-code-box" id='shareCode'></div>
                    <div class="share-code-text">
                      <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/share/icon-wechat.png" alt="">扫一扫
                    </div>
                  </div>
                </div>
              </div>
            </li>
                <li class="tool-item tool-item-size tool-active tool-item-reward">
                  <a class="tool-item-href" href="javascript:;" data-report-click='{"mod":"popu_830","spm":"1001.2101.3001.4237","dest":"","ab":"new"}'>
                    <img class="isdefault reward-bt" id="rewardBtNew" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/reward.png" alt="打赏">
                    <span class="count">打赏</span>
                  </a>
                  <div class="tool-hover-tip"><span class="text space">打赏</span></div>
                </li>
          <li class="tool-item tool-item-size tool-active is-more" id="is-more">
            <a class="tool-item-href">
              <img class="isdefault" style="margin-right:0px;display:block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/more.png" alt="">
              <span class="count"></span>
            </a>
            <div class="more-opt-box">
              <div class="mini-box">
                    <a class="tool-item-href" id="rewardBtNewHide" data-report-click='{"spm":"3001.4237","extra":"{\"type\":\"hide\"}"}'>
                      <img class="isdefault reward-bt" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/reward.png" alt="打赏">
                      <span class="count">打赏</span>
                    </a>
                <a class="tool-item-href" id="toolReportBtnHide">
                  <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png" alt="">
                  <span class="count">举报</span>
                </a>
              </div>
              <div class="normal-box">
                <a class="tool-item-href" id="toolReportBtnHideNormal">
                  <img class="isdefault" src="https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png" alt="">
                  <span class="count">举报</span>
                </a>
              </div>
            </div>
          </li>
        </ul>
      </div>
      <div class="toolbox-right">
            <div class="tool-directory">
                <a class="bt-columnlist-show"
                  data-id="9281458"
                  data-free="true"
                  data-description="分享本人学习Linux DRM (Direct Rendering Manager) 图形架构的经验总结，并以最简单的示例展示如何编写DRM应用程序和驱动程序，简单易懂，适合初学者。"
                  data-subscribe="false"
                  data-title="DRM (Direct Rendering Manager)"
                  data-img="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_224,w_224"
                  data-url="https://blog.csdn.net/hexiaolong2009/category_9281458.html"
                  data-sum="29"
                  data-people="1235"
                  data-price="0"
                  data-hotRank="0"
                  data-status="true"
                  data-oldprice="0"
                  data-join="false"
                  data-studyvip="false"
                  data-studysubscribe="false"
                  data-report-view='{"spm":"1001.2101.3001.6334","extend1":"专栏目录"}'
                  data-report-click='{"spm":"1001.2101.3001.6334","extend1":"专栏目录"}'>专栏目录</a>
          </div>
</div>
</div>
</div>
<script type=text/javascript crossorigin src="https://csdnimg.cn/release/phoenix/production/qrcode-7c90a92189.min.js"></script>
<script type="text/javascript" crossorigin src="https://g.csdnimg.cn/common/csdn-login-box/csdn-login-box.js"></script>
<script type="text/javascript" crossorigin src="https://g.csdnimg.cn/collection-box/2.1.2/collection-box.js"></script>
                <div class="first-recommend-box recommend-box ">
<div class="recommend-item-box type_blog clearfix" data-url="https://bill-huang.blog.csdn.net/article/details/128293518"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-128293518-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://bill-huang.blog.csdn.net/article/details/128293518"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://bill-huang.blog.csdn.net/article/details/128293518" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-128293518-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://bill-huang.blog.csdn.net/article/details/128293518"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-128293518-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-128293518-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux内存从0到1学习笔记<em>(</em>8.6 <em>DMA</em><em>-</em><em>BUF</em>简介）</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/huangyabin001" target="_blank"><span class="blog-title">从0到1，突破自己</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">12-12</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					6848
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://bill-huang.blog.csdn.net/article/details/128293518" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-128293518-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://bill-huang.blog.csdn.net/article/details/128293518"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-128293518-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-128293518-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">CPU指令系统通常只支持CPU（寄存器）<em>-</em>存储器，以及CPU<em>-</em>外设之间的数据传送，那么如果外设需要和存储器进行数据交换就必须经过CPU寄存器进行中转。很显然，中转会大大降低CPU的工作效率，浪费时间。因此，需要在外设和存储器之间开辟一个直接数据传输通道。如果这个通道的数据传输有其他硬件来完成，既可以加快传输效率，又可以减轻CPU对I/O的负载。</div>
			</a>
		</div>
	</div>
</div>
                </div>
            <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_commontools-829a4838ae.min.js" type="text/javascript" async></script>
              <div class="second-recommend-box recommend-box ">
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-140369236-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-140369236-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140369236-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140369236-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">缓冲区共享和同步<em>dma</em>_<em>buf</em> 之一</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/IT_Beijing_BIT" target="_blank"><span class="blog-title">IT_Beijing_BIT的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">07-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2323
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-140369236-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140369236"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140369236-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140369236-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> 子系统提供了用于跨多个设备驱动程序和子系统共享硬件 <em>(</em><em>DMA</em><em>)</em> 访问缓冲区以及同步异步硬件访问的框架。</div>
			</a>
		</div>
	</div>
</div>
              </div>
<a id="commentBox" name="commentBox"></a>
  <div id="pcCommentBox" class="comment-box comment-box-new2 unlogin-comment-box-new" style="display:none">
      <div class="unlogin-comment-model">
          <span class="unlogin-comment-tit">14&nbsp;条评论</span>
        <span class="unlogin-comment-text">您还未登录，请先</span>
        <span class="unlogin-comment-bt">登录</span>
        <span class="unlogin-comment-text">后发表或查看评论</span>
      </div>
  </div>
  <div class="blog-comment-box-new" style="display: none;">
        <h1>15 条评论</h1>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/talk2stars">
              <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                alt="talk2stars" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/talk2stars">
                      <span class="name ">talk2stars</span></a>
                    <span class="date" title="2025-07-28 04:02:39">2025.07.28</span>
                    <div class="new-comment">讲得真棒! 想问一下博主, dmabuf_fd用了 就不管了吗</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/tao475824827">
              <img src="https://profile-avatar.csdnimg.cn/598fb43a32504785b0d1f483429f379b_tao475824827.jpg!1"
                alt="tao475824827" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/tao475824827">
                      <span class="name ">taotao830</span></a>
                    <span class="date" title="2024-02-03 08:19:21">2024.02.03</span>
                    <div class="new-comment">作者有出版纸质书吗，感觉写的通俗易懂。</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/ningxai120">
              <img src="https://profile-avatar.csdnimg.cn/8cef88ca56144f54861c2c34b1f8e990_ningxai120.jpg!1"
                alt="ningxai120" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/ningxai120">
                      <span class="name ">linux呵呵</span></a>
                    <span class="date" title="2024-01-26 17:20:19">2024.01.26</span>
                    <div class="new-comment">ioctl(fd, 0, &amp;dmabuf_fd); 只能执行一次，然后 int fd = dma_buf_fd(dmabuf_exported, O_CLOEXEC);就返回负数了，这个fd可以在用户空间传递出去后，就不管了吗？</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/MeloSydneyUni">
              <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                alt="MeloSydneyUni" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/MeloSydneyUni">
                      <span class="name ">Melo__</span></a>
                    <span class="date" title="2023-12-07 10:13:47">2023.12.07</span>
                    <div class="new-comment">想问一下博主, mmap这种方式是否已经不推荐了 ? 我在内核源码里面很少能看到dmabuf 提供mmap的方式了 我不确定哈 想问一下</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/liangwei511">
              <img src="https://profile-avatar.csdnimg.cn/7b968b906fb1470aaedc82e7d033016a_liangwei511.jpg!1"
                alt="liangwei511" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/liangwei511">
                      <span class="name ">赛斯迪恩</span></a>
                    <span class="date" title="2023-07-28 10:00:54">2023.07.28</span>
                    <div class="new-comment">用户态如果要实现通过DMA BUF驱动分配任意字节的buffer，并拿到用户态去使用，共享等该怎么操作呢</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
          <li >
            <ul>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/MeloSydneyUni">
                    <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                      alt="MeloSydneyUni" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/MeloSydneyUni">
                            <span class="name ">Melo__</span><span class="text">回复</span><span class="nick-name">赛斯迪恩</span>
                          </a>
                          <span class="date" title="2023-12-07 10:12:40">2023.12.07</span>
                          <div class="new-comment">有现成的 udmabuf 研究下怎么用就行</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
            </ul>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/m0_56922987">
              <img src="https://profile-avatar.csdnimg.cn/eeb4518014a34d009b26a4313a3d6d22_m0_56922987.jpg!1"
                alt="m0_56922987" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/m0_56922987">
                      <span class="name ">德玛西亚最后的荣光</span></a>
                    <span class="date" title="2022-10-18 10:46:47">2022.10.18</span>
                    <div class="new-comment">讲得真棒</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/weixin_43461715">
              <img src="https://profile-avatar.csdnimg.cn/b883b13a6c8e441395b87a65e6787834_weixin_43461715.jpg!1"
                alt="weixin_43461715" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/weixin_43461715">
                      <span class="name ">陆雪棋</span></a>
                    <span class="date" title="2021-07-09 16:34:18">2021.07.09</span>
                    <div class="new-comment">实验加载了exporter.ko 之后，没有在/dev/下产生设备，是应为rootfs中没有编译进去udev工具吗？ 我尝试自己编译了一下，都失败了，请问您编译过吗？</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
          <li >
            <ul>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/hexiaolong2009">
                    <img src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1"
                      alt="hexiaolong2009" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/hexiaolong2009">
                            <span class="name ">何小龙</span><span class="text">回复</span><span class="nick-name">陆雪棋</span>
                          </a>
                          <span class="date" title="2021-07-09 23:23:55">2021.07.09</span>
                          <div class="new-comment">在 kernel menuconfig 中开启 devtmpfs 选项应该就可以了</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
            </ul>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/gangjian68">
              <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                alt="gangjian68" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/gangjian68">
                      <span class="name ">gangjian68</span></a>
                    <span class="date" title="2021-03-18 10:52:31">2021.03.18</span>
                    <div class="new-comment">首先通过 exporter 驱动的 ioctl() 获取到 dma-buf 的 fd，然后直接使用该 fd 做 mmap() 映射
===&gt;fd是从内核传上来的，这个fd属于这个进程吗？</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
          <li >
            <ul>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/hexiaolong2009">
                    <img src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1"
                      alt="hexiaolong2009" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/hexiaolong2009">
                            <span class="name ">何小龙</span><span class="text">回复</span><span class="nick-name">gangjian68</span>
                          </a>
                          <span class="date" title="2021-03-18 14:53:21">2021.03.18</span>
                          <div class="new-comment">属于</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                <li>
                  <a target="_blank" href="https://blog.csdn.net/gangjian68">
                    <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                      alt="gangjian68" class="avatar">
                  </a>
                  <div class="right-box">
                    <div class="new-info-box clearfix">
                      <div class="comment-top">
                        <div class="user-box">
                          <a class="name-href" target="_blank"  href="https://blog.csdn.net/gangjian68">
                            <span class="name ">gangjian68</span><span class="text">回复</span><span class="nick-name">gangjian68</span>
                          </a>
                          <span class="date" title="2021-03-18 11:08:47">2021.03.18</span>
                          <div class="new-comment">解决了，我看了一下install_fd，它传进去的参数是current-&gt;file，这个fd就属于当前进程了。</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
            </ul>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/xiashaohua">
              <img src="https://profile-avatar.csdnimg.cn/dafc5356b78a4591965cc773b94ece26_xiashaohua.jpg!1"
                alt="xiashaohua" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/xiashaohua">
                      <span class="name ">维民所止1226</span></a>
                    <span class="date" title="2021-02-03 15:53:36">2021.02.03</span>
                    <div class="new-comment">#:/local/mnt/workspace/qemu/linux-4.14.217/drivers/mmap-test$ make
#arm-linux-gnueabi-gcc test.c -o mmap-tst
#:/local/mnt/workspace/qemu/linux-4.14.217/drivers/mmap-test$ cp mmap4.ko ../../../busybox/root/rootfs/mmap-test/
#:/local/mnt/workspace/qemu/linux-4.14.217/drivers/mmap-test$ cp mmap-tst ../../../busybox/root/rootfs/mmap-test/
#:/local/mnt/workspace/qemu/busybox/root$ ps -a
#:/local/mnt/workspace/qemu/busybox/root$ kill 26704
#:/local/mnt/workspace/qemu/busybox/root$ ./mount-rootfs.sh
#== equal to mkdir etc ; cd /etc ; mkdir init.d ; cd init.d ; touch rcS ;chmod a+x rcS
#:/local/mnt/workspace/qemu/busybox/root$ ./boot.sh
#== equal to
#qemu-system-arm -M vexpress-a9 \
#    -m 512M \
#    -kernel zImage \
#    -dtb vexpress-v2p-ca9.dtb \
#    -nographic \
#    -append &quot;root=/dev/mmcblk0 rw console=ttyAMA0&quot; \
#    -sd rootfs.ext3
# in the qemu
# # mount -t proc none /proc
#ls
# cd mmap-test
#/mmap-test # insmod mmap4.ko
#lsmod

#/mmap-test # ./mmap-tst</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/xiashaohua">
              <img src="https://profile-avatar.csdnimg.cn/dafc5356b78a4591965cc773b94ece26_xiashaohua.jpg!1"
                alt="xiashaohua" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/xiashaohua">
                      <span class="name ">维民所止1226</span></a>
                    <span class="date" title="2021-02-03 15:53:15">2021.02.03</span>
                    <div class="new-comment">使用方法：


#:/local/mnt/workspace/qemu/linux-4.14.217/drivers/mmap-test$ make
#arm-linux-gnueabi-gcc test.c -o mmap-tst
#:/local/mnt/workspace/qemu/linux-4.14.217/drivers/mmap-test$ cp mmap4.ko ../../../busybox/root/rootfs/mmap-test/
#:/local/mnt/workspace/qemu/linux-4.14.217/drivers/mmap-test$ cp mmap-tst ../../../busybox/root/rootfs/mmap-test/
#:/local/mnt/workspace/qemu/busybox/root$ ps -a
#:/local/mnt/workspace/qemu/busybox/root$ kill 26704
#:/local/mnt/workspace/qemu/busybox/root$ ./mount-rootfs.sh
#== equal to mkdir etc ; cd /etc ; mkdir init.d ; cd init.d ; touch rcS ;chmod a+x rcS
#:/local/mnt/workspace/qemu/busybox/root$ ./boot.sh
#== equal to
#qemu-system-arm -M vexpress-a9 \
#    -m 512M \
#    -kernel zImage \
#    -dtb vexpress-v2p-ca9.dtb \
#    -nographic \
#    -append &quot;root=/dev/mmcblk0 rw console=ttyAMA0&quot; \
#    -sd rootfs.ext3
# in the qemu
# # mount -t proc none /proc
#ls
# cd mmap-test
#/mmap-test # insmod mmap4.ko
#lsmod

#/mmap-test # ./mmap-tst</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
    </div>
              <div class="recommend-box insert-baidu-box recommend-box-style ">
                <div class="recommend-item-box no-index" style="display:none"></div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596761"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~activity-2-102596761-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~baidujs_baidulandingword~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596761"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596761" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~activity-2-102596761-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~baidujs_baidulandingword~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596761"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7Eactivity-2-102596761-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7Eactivity-2-102596761-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（二） &mdash;&mdash; kmap / <em>vmap</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					3万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596761" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~activity-2-102596761-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~baidujs_baidulandingword~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596761"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7Eactivity-2-102596761-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7Eactivity-2-102596761-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在上一篇《最简单的 <em>dma</em><em>-</em><em>buf</em> 驱动程序》中，我们学习了编写 <em>dma</em><em>-</em><em>buf</em> 驱动程序的三个基本步骤，即 <em>dma</em>_<em>buf</em>_ops 、 <em>dma</em>_<em>buf</em>_export_info、 <em>dma</em>_<em>buf</em>_export<em>(</em><em>)</em>。在本篇中，我们将在 exporter<em>-</em>dummy 驱动的基础上，对其 <em>dma</em>_<em>buf</em>_ops 的 kmap / <em>vmap</em> 接口进行扩展，以此来演示这两个接口的使用方法。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://zhugeyifan.blog.csdn.net/article/details/154238790"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-154238790-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~YuanLiJiHua~Position","dest":"https://zhugeyifan.blog.csdn.net/article/details/154238790"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://zhugeyifan.blog.csdn.net/article/details/154238790" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-154238790-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~YuanLiJiHua~Position","dest":"https://zhugeyifan.blog.csdn.net/article/details/154238790"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-154238790-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-154238790-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">【<em>内存管理</em>】深入理解内存映射（Memory Mapping）与<em>mmap</em>：实现高效零拷贝的<em>DMA</em>缓冲区共享</div>
					<div class="tag">最新发布</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/Ivan804638781" target="_blank"><span class="blog-title">诸葛一帆丶的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-01</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1523
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://zhugeyifan.blog.csdn.net/article/details/154238790" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-154238790-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~YuanLiJiHua~Position","dest":"https://zhugeyifan.blog.csdn.net/article/details/154238790"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-154238790-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-154238790-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">内存映射<em>(</em><em>mmap</em><em>)</em>是操作系统提供的将文件或设备直接映射到进程虚拟地址空间的机制，允许进程像访问内存一样访问文件内容或设备内存区域。文章详细介绍了<em>mmap</em>系统调用的工作机制，重点阐述了将内核<em>DMA</em>缓冲区映射到用户空间的完整流程，包括驱动端<em>DMA</em>缓冲区分配、<em>mmap</em>文件操作实现以及用户空间映射方法。通过remap_pfn_range或<em>dma</em>_<em>mmap</em>_coherent函数建立物理页到用户空间的映射，实现零拷贝数据传输。相比传统read/write方式，<em>mmap</em>消除了数据拷贝开销，减少了系统调用次数，提升了性能</div>
			</a>
		</div>
	</div>
</div>
		<dl id="recommend-item-box-tow" class="recommend-item-box type_blog clearfix">
			
		</dl>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/huntenganwei/article/details/147550084"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-147550084-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/huntenganwei/article/details/147550084"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/huntenganwei/article/details/147550084" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-147550084-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/huntenganwei/article/details/147550084"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-147550084-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-147550084-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em>_<em>buf</em>学习记录之二核心接口</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/huntenganwei" target="_blank"><span class="blog-title">huntenganwei的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">04-27</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1114
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/huntenganwei/article/details/147550084" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-147550084-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/huntenganwei/article/details/147550084"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-147550084-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-147550084-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">`DEFINE_<em>DMA</em>_<em>BUF</em>_EXPORT_INFO`：定义并初始化 `<em>dma</em>_<em>buf</em>_export_info` 结构体。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/139242952"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-139242952-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/139242952"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/139242952" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-139242952-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/139242952"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-139242952-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-139242952-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">linux 之<em>dma</em>_<em>buf</em> <em>(</em>4<em>)</em><em>-</em> <em>mmap</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/z20230508" target="_blank"><span class="blog-title">z20230508的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">05-28</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1721
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/139242952" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-139242952-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/139242952"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-139242952-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-139242952-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">前面几篇都是在 kernel space 对 <em>dma</em><em>-</em><em>buf</em> 进行访问的，本篇我们将一起来学习，如何在 user space 访问 <em>dma</em><em>-</em><em>buf</em>。当然，user space 访问 <em>dma</em><em>-</em><em>buf</em> 也属于 CPU Access 的一种。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596772"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-6-102596772-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596772" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-6-102596772-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-6-102596772-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-6-102596772-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（三） &mdash;&mdash; map attachment</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					3万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596772" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-6-102596772-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-6-102596772-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-6-102596772-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在上一篇《kmap/<em>vmap</em>》中，我们学习了如何使用 CPU 在 kernel 空间访问 <em>dma</em><em>-</em><em>buf</em> 物理内存，但如果使用CPU直接去访问 memory，那么性能会大大降低。因此，<em>dma</em><em>-</em><em>buf</em> 在内核中出现频率最高的还是它的 <em>dma</em>_<em>buf</em>_attach<em>(</em><em>)</em> 和 <em>dma</em>_<em>buf</em>_map_attachment<em>(</em><em>)</em> 接口。本篇我们就一起来学习如何通过这两个 API 来实现 <em>DMA</em> 硬件对 <em>dma</em><em>-</em><em>buf</em> 物理内存的访问。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_chatgpt clearfix" data-url="https://wenku.csdn.net/answer/3xk6xy573a"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/3xk6xy573a"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/answer/3xk6xy573a" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/3xk6xy573a"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em>_<em>buf</em>的源码在哪个位置</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block">10-31</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/answer/3xk6xy573a" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/3xk6xy573a"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">我们被要求查找<em>dma</em>_<em>buf</em>源码在Linux内核中的位置。根据Linux内核源码的结构，<em>dma</em>_<em>buf</em>相关...[^1]: <em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（<em>四</em>） &mdash;&mdash; <em>mmap</em> [^2]: <em>dma</em>_<em>buf</em>学习记录之一基础知识 [^3]: linux 之<em>dma</em>_<em>buf</em> <em>(</em>1<em>)</em><em>-</em> <em>dma</em>_<em>buf</em> 的初步介绍</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/weixin_45449806/article/details/140582909"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-8-140582909-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"8","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/weixin_45449806/article/details/140582909"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/weixin_45449806/article/details/140582909" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-8-140582909-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"8","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/weixin_45449806/article/details/140582909"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-8-140582909-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-8-140582909-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Soc 离屏渲染优化 <em>-</em> 传输优化</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_45449806" target="_blank"><span class="blog-title">皓瑞的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">07-21</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1250
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/weixin_45449806/article/details/140582909" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-8-140582909-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"8","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/weixin_45449806/article/details/140582909"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-8-140582909-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-8-140582909-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在组装 PC 整机时, 我们会说 CPU XXX 型号, DDR XXX G, 显卡 XXX 型号显存 XXX G;这是一个很自然而言的想法, 那么业界的大佬是否也有这种想法;既然不同的硬件单元的内存实际上是连接在一起的.在序中我曾提及到如果不需要将渲染的结果进行输出, 这条数据链路的传输可以由。这固然解决了一些数据共享问题, 但更多是站在。总线这个角度, 这个角色是谁的呢?导入纹理的, 因为这种方式不通用不可跨平台.是可以的, 这里我们不展开;纹理是降低开销, 提升性能的有效手段.那么是不是可以共享呢?</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/relax33/article/details/128319124"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-9-128319124-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"9","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/relax33/article/details/128319124" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-9-128319124-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"9","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-9-128319124-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-9-128319124-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">ION to SMMU</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/relax33" target="_blank"><span class="blog-title">relax33的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">12-14</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1484
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/relax33/article/details/128319124" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-9-128319124-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"9","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-9-128319124-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-9-128319124-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">ION <em>DMA</em><em>-</em><em>BUF</em> IOMMU</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/langjian2012/article/details/144420600"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-10-144420600-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"10","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/langjian2012/article/details/144420600"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/langjian2012/article/details/144420600" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-10-144420600-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"10","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/langjian2012/article/details/144420600"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-10-144420600-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-10-144420600-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">memory内存分类</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/langjian2012" target="_blank"><span class="blog-title">langjian2012的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">12-12</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					474
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/langjian2012/article/details/144420600" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-10-144420600-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"10","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/langjian2012/article/details/144420600"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-10-144420600-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-10-144420600-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>(</em>adb shell dumpsys meminfo x<em>)</em>堆内存用于存储对象实例和静态变量</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_wk_aigc_column clearfix" data-url="https://wenku.csdn.net/column/7z3sy0xjbm"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/7z3sy0xjbm"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/column/7z3sy0xjbm" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/7z3sy0xjbm"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">嵌入式Linux<em>内存管理</em>机制剖析：MMU工作原理、<em>mmap</em>应用与内存泄漏防范的4大要点</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block"></span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/column/7z3sy0xjbm" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/7z3sy0xjbm"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">!...# 1. 嵌入式Linux<em>内存管理</em>概述  在嵌入式Linux系统中，<em>内存管理</em>是保障系统稳定性与性能的核心机制。...<em>内存管理</em>的核心组件包括MMU（<em>内存管理</em>单元）、页表机制、<em>mmap</em>系统调用及动态内存分配器等。这些机制协同工</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596744"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596744-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596744" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596744-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（一） &mdash;&mdash; 最简单的 <em>dma</em><em>-</em><em>buf</em> 驱动程序</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					8万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596744" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596744-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">如果你和我一样，是一位从事Android多媒体底层开发的工程师，那么你对 <em>dma</em><em>-</em><em>buf</em> 这个词语一定不会陌生，因为不管是 Video、Camera 还是 Display、GPU，它们的<em>buf</em>fer都来自于ION，而 ION 正是基于 <em>dma</em><em>-</em><em>buf</em> 实现的。

假如你对 <em>dma</em><em>-</em><em>buf</em> 的理解并不深刻，又期望找个时间来彻底公关一下，那么很高兴，这几篇文章一定能让你对 <em>dma</em><em>-</em><em>buf</em> 有个更深入、更透彻的理解。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596845"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-13-102596845-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596845"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596845" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-13-102596845-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596845"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-13-102596845-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-13-102596845-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（七） &mdash;&mdash; alloc page 版本</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">01-12</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596845" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-13-102596845-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596845"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-13-102596845-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-13-102596845-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在前面的 <em>dma</em><em>-</em><em>buf</em> 系列文章中，exporter 所分配的内存都是通过 kzalloc<em>(</em><em>)</em> 来分配的。本篇我们换个方式，使用 alloc_page<em>(</em><em>)</em> 来分配内存。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/qq_18998145/article/details/99406944"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-99406944-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/qq_18998145/article/details/99406944"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/qq_18998145/article/details/99406944" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-99406944-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/qq_18998145/article/details/99406944"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-99406944-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-99406944-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DMA</em>_<em>BUF</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/qq_18998145" target="_blank"><span class="blog-title">LIEY</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">08-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2990
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/qq_18998145/article/details/99406944" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-99406944-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/qq_18998145/article/details/99406944"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-99406944-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-99406944-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">考虑这样一种场景，摄像头采集的视频数据需要送到GPU中进行编码、显示。负责数据采集和编码的模块是Linux下不同的驱动设备，将采集设备中的数据送到编码设备中 需要一种方法。最简单的方法可能就是进行一次内存拷贝，但是我们这里需要寻求一种免拷贝的通用方法。<em>dma</em>_<em>buf</em>是内核中一个独立的子系统，可以让不同设备、子系统之间进行内存共享的统一机制。

<em>DMA</em>_<em>BUF</em>框架下主要有两个角色对象，一个是expo...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/52139585"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-15-52139585-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/52139585"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/52139585" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-15-52139585-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/52139585"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-15-52139585-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-15-52139585-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux 内存映射函数 <em>mmap</em>（）函数详解</div>
					<div class="tag">热门推荐</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/yangle4695" target="_blank"><span class="blog-title">yangle4695的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">08-07</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					11万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/52139585" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-15-52139585-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/52139585"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-15-52139585-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-15-52139585-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>mmap</em>将一个文件或者其它对象映射进内存。文件被映射到多个页上，如果文件的大小不是所有页的大小之和，最后一个页不被使用的空间将会清零。<em>mmap</em>在用户空间映射调用系统中作用很大。
头文件 
函数原型
void* <em>mmap</em><em>(</em>void* start,size_t length,int prot,int flags,int fd,off_t offset<em>)</em>;
int munmap<em>(</em>void* st</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/weixin_39592381/article/details/133790159"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-133790159-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39592381/article/details/133790159"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/weixin_39592381/article/details/133790159" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-133790159-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39592381/article/details/133790159"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-133790159-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-133790159-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DMA</em><em>-</em><em>BUF</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_39592381" target="_blank"><span class="blog-title">weixin_39592381的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">10-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					752
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/weixin_39592381/article/details/133790159" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-133790159-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_39592381/article/details/133790159"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-133790159-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-133790159-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">模块初始化函数，注册<em>dma</em>_<em>buf</em>_fs，初始化db_list，初始化debugfs。初始化一个attach节点，并把它加入到<em>dmabuf</em>的attachments列表中。成功会返回&amp;<em>dma</em>_<em>buf</em>的指针，失败会返回一个负数（通过ERR_PTR包装）。创建一个<em>dmabuf</em>，并把它关联到一个anon file上，以便暴露这块内存。锁住一块<em>dmabuf</em>。从系统中获取一个可用的fd，并把它跟<em>dmabuf</em><em>-</em>&gt;file绑定起来。调用用户定义的unmap_<em>dma</em>_<em>buf</em>回调。调用用户定义的map_<em>dma</em>_<em>buf</em>回调。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/abc3240660/article/details/81942190"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-81942190-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/abc3240660/article/details/81942190" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-81942190-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-81942190-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-81942190-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux内核笔记之<em>DMA</em>_<em>BUF</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/abc3240660" target="_blank"><span class="blog-title">abc3240660的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">08-22</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					8687
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/abc3240660/article/details/81942190" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-81942190-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-81942190-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-17-81942190-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">
      
        Linux内核笔记之<em>DMA</em>_<em>BUF</em>
        Apr 18, 2018
      
      
      
        
  <em>DMA</em>_<em>BUF</em>    
      需求背景
      概述
      <em>dma</em><em>-</em><em>buf</em>实现
      运作流程
      Importer驱动实例剖析
      Export驱动实例编写
    
  


<em>内存管理</em>...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/7940330"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-18-7940330-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/7940330" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-18-7940330-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-18-7940330-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-18-7940330-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">linux之<em>DMA</em><em>-</em><em>BUF</em> API使用指南</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/crazyjiang" target="_blank"><span class="blog-title">crazyjiang的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">09-04</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/7940330" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-18-7940330-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-18-7940330-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-18-7940330-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>DMA</em><em>-</em><em>BUF</em> API使用指南
by JHJ<em>(</em>jianghuijun211@gmail.com<em>)</em>
转载出自：http://blog.csdn.net/crazyjiang
本文将会告诉驱动开发者什么是<em>dma</em><em>-</em><em>buf</em>共享缓冲区接口，如何作为一个生产者及消费者使用共享缓冲区。
任何一个设备驱动想要使用<em>DMA</em>共享缓冲区，就必须为缓冲区的生产者或者消费者。
如果驱动A想用驱动B创建的缓冲区，那么</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_chatgpt clearfix" data-url="https://wenku.csdn.net/answer/2qr6q87w5b"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.19","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/answer/2qr6q87w5b" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.19","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'  data-report-query='spm=1001.2101.3001.6650.19&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block">05-13</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/answer/2qr6q87w5b" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.19","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-2qr6q87w5b-blog-102596791.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382488789_95616\"}","dist_request_id":"1766382488789_95616","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/2qr6q87w5b"}'  data-report-query='spm=1001.2101.3001.6650.19&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-2qr6q87w5b-blog-102596791.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">### 什么是 <em>DMA</em><em>-</em><em>BUF</em>？  <em>DMA</em><em>-</em><em>BUF</em> 是 Linux 内核中的一个框架，用于跨子系统的缓冲区共享。它允许不同的硬件子系统（如 GPU、DSP 或其他外设）之间高效地共享内存数据，从而减少不必要的复制操作并提高性能[^5]。  <em>-</em><em>-</em><em>-</em>...</div>
			</a>
		</div>
	</div>
</div>
              </div>
<div class="blog-footer-bottom" style="margin-top:10px;"></div>
<script src="https://g.csdnimg.cn/common/csdn-footer/csdn-footer.js" data-isfootertrack="false" type="text/javascript"></script>
<script type="text/javascript">
    window.csdn.csdnFooter.options = {
        el: '.blog-footer-bottom',
        type: 2
    }
</script>          </main>
<aside class="blog_container_aside ">
<div id="asideProfile" class="aside-box active">
    <div class="profile-intro d-flex">
        <div class="avatar-box d-flex justify-content-center flex-column">
            <a href="https://blog.csdn.net/hexiaolong2009" target="_blank" data-report-click='{"mod":"popu_379","spm":"3001.4121","dest":"https://blog.csdn.net/hexiaolong2009","ab":"new"}'>
                <img src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1" class="avatar_pic">
            </a>
        </div>
        <div class="user-info d-flex flex-column profile-intro-name-box">
            <div class="profile-intro-name-boxTop">
                <a href="https://blog.csdn.net/hexiaolong2009" target="_blank" class="" id="uid" title="何小龙" data-report-click='{"mod":"popu_379","spm":"3001.4122","dest":"https://blog.csdn.net/hexiaolong2009","ab":"new"}'>
                    <span class="name" username="hexiaolong2009">何小龙</span>
                </a>
            </div>
            <div class="profile-intro-name-boxFooter-new">
              <p class="profile-intro-name-leve">
                <span>
                  博客等级
                </span>
                  <img class="level" src="https://csdnimg.cn/identity/blog6.png">
              </p>
                <span class="profile-intro-name-years" title="已加入 CSDN 16年">码龄16年</span>
               
            </div>
        </div>
    </div>
    <div class="profile-intro-Identity-information">
        <p class="profile-information-box">
          <img class="information-img" data-report-click='{"spm":"3001.4296"}' src="https://i-operation.csdnimg.cn/images/586260c6ecd54b20be60ced2d94df1d8.png" alt="">
          <span>领域专家: 嵌入式与硬件开发技术领域</span>
        </p>

    </div>
    <div class="profile-intro-rank-information">
      <dl>
        <a href="https://blog.csdn.net/hexiaolong2009" data-report-click='{"mod":"1598321000_001","spm":"3001.4310"}' data-report-query="t=1">  
            <dd><span >84</span></dd>
            <dt>原创</dt>
        </a>
      </dl>
       <dl title="1941">
        <dd>1941</dd>
        <dt>点赞</dt>
      </dl>
       <dl title="4922">
        <dd>4922</dd>
        <dt>收藏</dt>
      </dl>
      <dl id="fanBox" title="3270">
        <dd><span id="fan">3270</span></dd>
        <dt>粉丝</dt>
      </dl>
    </div>
    <div class="profile-intro-name-boxOpration">
        <div class="opt-letter-watch-box"> 
            <a class="personal-watch bt-button" id="btnAttent" >关注</a>  
        </div>
        <div class="opt-letter-watch-box">
        <a rel="nofollow" class="bt-button personal-letter" href="https://im.csdn.net/chat/hexiaolong2009" target="_blank" rel="noopener">私信</a>
        </div>
    </div>
</div>



<div id="asideHotArticle" class="aside-box">
	<h3 class="aside-title">热门文章</h3>
	<div class="aside-content">
		<ul class="hotArticle-list">
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/83720940" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/83720940","ab":"new"}'>
				DRM（Direct Rendering Manager）学习简介
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">121270</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596744" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/102596744","ab":"new"}'>
				dma-buf 由浅入深（一） —— 最简单的 dma-buf 驱动程序
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">83296</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/83721242" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/83721242","ab":"new"}'>
				最简单的DRM应用程序 （single-buffer）
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">64173</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/89810355" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/89810355","ab":"new"}'>
				DRM 驱动程序开发（开篇）
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">48022</span>
                </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/hexiaolong2009/article/details/79319512" target="_blank"  data-report-click='{"mod":"popu_541","spm":"3001.4139","dest":"https://blog.csdn.net/hexiaolong2009/article/details/79319512","ab":"new"}'>
				LCD显示异常分析——撕裂(tear effect)
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">42349</span>
                </a>
			</li>
		</ul>
	</div>
</div>
<div id="asideCategory" class="aside-box aside-box-column ">
    <h3 class="aside-title">分类专栏</h3>
    <div class="aside-content" id="aside-content">
        <ul>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9281458.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9281458.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        DRM (Direct Rendering Manager)
                    </span>
                </a>
                <span class="special-column-num">29篇</span>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_10331964.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_10331964.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756757.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        Linux Graphics 周刊
                    </span>
                </a>
                <span class="special-column-num">10篇</span>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9813335.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9813335.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        Wayland
                    </span>
                </a>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9705063.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9705063.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        GPU
                    </span>
                </a>
                <span class="special-column-num">6篇</span>
            </li>
            <li>
                <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_7583191.html" data-report-click='{"mod":"popu_537","spm":"3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_7583191.html","ab":"new"}'>
                    <div class="special-column-bar "></div>
                    <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                    <span class="title oneline">
                        Android
                    </span>
                </a>
                <span class="special-column-num">4篇</span>
            </li>
        </ul>
    </div>
</div>
  <div class="article-previous" id="article-previous">
      <dl data-report-click='{"spm":"3001.10752","extend1":"上一篇"}' data-report-view='{"spm":"3001.10752","extend1":"上一篇"}'>
          <dt>
              上一篇：
          </dt>
          <dd>
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（三） —— map attachment
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（五） —— File
            </a>
          </dd>
      </dl>
  </div>
<div id="asideHotArticle" class="aside-box">
	<h3 class="aside-title">大家在看</h3>
	<div class="aside-content">
		<ul class="hotArticle-list">
			<li>
				<a href="https://blog.csdn.net/qq_63718344/article/details/156148764" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/qq_63718344/article/details/156148764","strategy":"202_1052723-3681430_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/qq_63718344/article/details/156148764","strategy":"202_1052723-3681430_RCMD","ab":"new"}'>
				深入理解Linux GPIO子系统
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">50</span>
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/2501_91798322/article/details/156148900" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/2501_91798322/article/details/156148900","strategy":"202_1052723-3681424_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/2501_91798322/article/details/156148900","strategy":"202_1052723-3681424_RCMD","ab":"new"}'>
				Transformer与CNN图像分类对比
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/qq_39980997/article/details/156115380" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/qq_39980997/article/details/156115380","strategy":"202_1052723-3681432_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/qq_39980997/article/details/156115380","strategy":"202_1052723-3681432_RCMD","ab":"new"}'>
				游戏化教学激发键盘指法兴趣
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">425</span>
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/qkh1234567/article/details/156146565" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/qkh1234567/article/details/156146565","strategy":"202_1052723-3681410_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/qkh1234567/article/details/156146565","strategy":"202_1052723-3681410_RCMD","ab":"new"}'>
				大模型面试高频题解析
        </a>
			</li>
			<li>
				<a href="https://blog.csdn.net/silver90239/article/details/156049522" target="_blank"  data-report-click='{"spm":"3001.10093","dest":"https://blog.csdn.net/silver90239/article/details/156049522","strategy":"202_1052723-3681396_RCMD","ab":"new"}' data-report-view='{"spm":"3001.10093","dest":"https://blog.csdn.net/silver90239/article/details/156049522","strategy":"202_1052723-3681396_RCMD","ab":"new"}'>
				Shell正则与文本处理精要
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					<span class="read">758</span>
        </a>
			</li>
		</ul>
	</div>
</div>
<div id="asideArchive" class="aside-box" style="display:block!important; width:300px;">
    <h3 class="aside-title">最新文章</h3>
    <div class="aside-content">
        <ul class="inf_list clearfix">
            <li class="clearfix">
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/111876034" target="_blank" data-report-click='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/111876034","ab":"left"}' data-report-view='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/111876034","ab":"left"}'>LWN 翻译：Atomic Mode Setting 设计简介（下）</a>
            </li>
            <li class="clearfix">
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/112498800" target="_blank" data-report-click='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/112498800","ab":"left"}' data-report-view='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/112498800","ab":"left"}'>Linux Graphics 周刊（第 9 期）</a>
            </li>
            <li class="clearfix">
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/107661938" target="_blank" data-report-click='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/107661938","ab":"left"}' data-report-view='{"mod":"popu_382","spm":"3001.4136","dest":"https://blog.csdn.net/hexiaolong2009/article/details/107661938","ab":"left"}'>LWN 翻译：Atomic Mode Setting 设计简介（上）</a>
            </li>
        </ul>
        <div class="archive-bar"></div>
        <div class="archive-box">
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2021&amp;month=02" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2021&amp;month=02"}'><span class="year">2021年</span><span class="num">2篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2020&amp;month=12" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2020&amp;month=12"}'><span class="year">2020年</span><span class="num">28篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2019&amp;month=11" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2019&amp;month=11"}'><span class="year">2019年</span><span class="num">15篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2018&amp;month=12" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2018&amp;month=12"}'><span class="year">2018年</span><span class="num">7篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2015&amp;month=04" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2015&amp;month=04"}'><span class="year">2015年</span><span class="num">12篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2014&amp;month=03" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2014&amp;month=03"}'><span class="year">2014年</span><span class="num">5篇</span></a></div>
                <div class="archive-list-item"><a href="https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2013&amp;month=12" target="_blank" data-report-click='{"mod":"popu_538","spm":"3001.4138","ab":"new","dest":"https://blog.csdn.net/hexiaolong2009?type=blog&amp;year=2013&amp;month=12"}'><span class="year">2013年</span><span class="num">28篇</span></a></div>
        </div>
    </div>
</div>
  <div class="starmap-box box3 aside-box" data-spm='3001.11253' data-id='gpu_img_ace_step' data-utm-source='side' data-report-view='{"spm":"3001.11253","extra":{"openMirrorId":"gpu_img_ace_step"}}' data-report-click='{"spm":"3001.11253","extra":{"openMirrorId":"gpu_img_ace_step"}}'>
    <div class="img-box">
      <img src="https://csdn-665-inscode.s3.cn-north-1.jdcloud-oss.com/image/cover/gpu_img_ace_step.png/middle" alt="ACE-Step">
      <div class="img-tag">
        AI算力推荐
      </div>
    </div>
    <div class="info-box">
      <p class="title">ACE-Step</p>
      <p class="desc" title="ACE-Step是由中国团队阶跃星辰（StepFun）与ACE Studio联手打造的开源音乐生成模型。 它拥有3.5B参数量，支持快速高质量生成、强可控性和易于拓展的特点。 最厉害的是，它可以生成多种语言的歌曲，包括但不限于中文、英文、日文等19种语言">ACE-Step是由中国团队阶跃星辰（StepFun）与ACE Studio联手打造的开源音乐生成模型。 它拥有3.5B参数量，支持快速高质量生成、强可控性和易于拓展的特点。 最厉害的是，它可以生成多种语言的歌曲，包括但不限于中文、英文、日文等19种语言</p>
      <div class="tag-box">
            <div class="tag-item">音乐合成</div>
            <div class="tag-item">ACE-Step</div>
      </div>
    </div>
    <div class="operate-box">
      <button class="btn-go-mall" data-spm='3001.11296'>镜像市场</button>
      <button class="btn-go-deploy">一键部署</button>
    </div>
  </div>
    <!-- 详情页显示目录 -->
<!--文章目录-->
<div id="asidedirectory" class="aside-box">
    <div class='groupfile groupfile-active' id="directory">
        <h3 class="aside-title">目录</h3>
        <div class="align-items-stretch group_item" id="align-items-stretch">
            <div class="pos-box">
            <div class="scroll-box">
                <div class="toc-box"></div>
            </div>
            </div>
        </div>
          <p class="flexible-btn-new active" id="flexible-btn-groupfile" data-report-click='{"spm":"3001.10780","strategy":"展开全部"}' data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch"><span class="text">展开全部</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-bot-White.png" alt=""></p>
          <p class="flexible-btn-new-close active" data-report-click='{"spm":"3001.10780","strategy":"收起"}'  data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch"><span class="text">收起</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-top-White.png" alt=""></p>
    </div>
</div>
<div class="gitcode-qc-left-box aside-box" data-report-click='{"spm":"3001.11256", "extra":"{\"position\":\"left\"}"}'></div>
</aside>
<script>
	$("a.flexible-btn").click(function(){
		$(this).parents('div.aside-box').removeClass('flexible-box');
		$(this).parents("p.text-center").remove();
	})
</script>
<script type="text/javascript"  src="https://g.csdnimg.cn/user-tooltip/2.7/user-tooltip.js"></script>
<script type="text/javascript"  src="https://g.csdnimg.cn/user-medal/2.0.0/user-medal.js"></script>        </div>
<div class="recommend-right align-items-stretch clearfix" id="rightAside" data-type="recommend">
    <aside class="recommend-right_aside">
              <div class="rightside-fixed-hide">
  <div class="starmap-box box3 aside-box" data-spm='3001.11253' data-id='gpu_img_ace_step' data-utm-source='side' data-report-view='{"spm":"3001.11253","extra":{"openMirrorId":"gpu_img_ace_step"}}' data-report-click='{"spm":"3001.11253","extra":{"openMirrorId":"gpu_img_ace_step"}}'>
    <div class="img-box">
      <img src="https://csdn-665-inscode.s3.cn-north-1.jdcloud-oss.com/image/cover/gpu_img_ace_step.png/middle" alt="ACE-Step">
      <div class="img-tag">
        AI算力推荐
      </div>
    </div>
    <div class="info-box">
      <p class="title">ACE-Step</p>
      <p class="desc" title="ACE-Step是由中国团队阶跃星辰（StepFun）与ACE Studio联手打造的开源音乐生成模型。 它拥有3.5B参数量，支持快速高质量生成、强可控性和易于拓展的特点。 最厉害的是，它可以生成多种语言的歌曲，包括但不限于中文、英文、日文等19种语言">ACE-Step是由中国团队阶跃星辰（StepFun）与ACE Studio联手打造的开源音乐生成模型。 它拥有3.5B参数量，支持快速高质量生成、强可控性和易于拓展的特点。 最厉害的是，它可以生成多种语言的歌曲，包括但不限于中文、英文、日文等19种语言</p>
      <div class="tag-box">
            <div class="tag-item">音乐合成</div>
            <div class="tag-item">ACE-Step</div>
      </div>
    </div>
    <div class="operate-box">
      <button class="btn-go-mall" data-spm='3001.11296'>镜像市场</button>
      <button class="btn-go-deploy">一键部署</button>
    </div>
  </div>
        </div>
        <div id="recommend-right" >
          <div class='flex-column aside-box groupfile groupfile-active ' id="groupfile">
              <div class="groupfile-div">
              <h3 class="aside-title">目录</h3>
              <div class="align-items-stretch group_item" id="align-items-stretch-right">
                  <div class="pos-box">
                      <div class="scroll-box">
                          <div class="toc-box"></div>
                      </div>
                  </div>
              </div>
                <p class="flexible-btn-new" id="flexible-btn-groupfile" data-report-click='{"spm":"3001.10782","strategy":"展开全部"}' data-traigger="true" data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch-right"><span class="text">展开全部</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-bot-White.png" alt=""></p>
                <p class="flexible-btn-new-close close" data-report-click='{"spm":"3001.10782","strategy":"收起"}' data-traigger="true"  data-minheight="117px" data-maxheight="446px" data-fbox="#align-items-stretch-right"><span class="text">收起</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-top-White.png" alt=""></p>
              </div>
          </div>
          <div class="gitcode-qc-right-box aside-box" data-report-click='{"spm":"3001.11256", "extra":"{\"position\":\"right\"}"}'></div>
  <div class="article-previous" id="article">
      <dl data-report-click='{"spm":"3001.10752","extend1":"上一篇"}' data-report-view='{"spm":"3001.10752","extend1":"上一篇"}'>
          <dt>
              上一篇：
          </dt>
          <dd>
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（三） —— map attachment
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（五） —— File
            </a>
          </dd>
      </dl>
  </div>
          <div class='aside-box kind_person d-flex flex-column flexible-box-new' >
                  <h3 class="aside-title">分类专栏</h3>
                  <div class="align-items-stretch kindof_item" id="kind_person_column">
                      <div class="aside-content" id="aside-content-column">
                          <ul>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9281458.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9281458.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/blog_column_migrate/c03d60f8b24740d58f94a787024155be.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          DRM (Direct Rendering Manager)
                                      </span>
                                  </a>
                                  <span class="special-column-num">29篇</span>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_10331964.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_10331964.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756757.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          Linux Graphics 周刊
                                      </span>
                                  </a>
                                  <span class="special-column-num">10篇</span>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9813335.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9813335.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          Wayland
                                      </span>
                                  </a>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_9705063.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_9705063.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          GPU
                                      </span>
                                  </a>
                                  <span class="special-column-num">6篇</span>
                              </li>
                              <li>
                                  <a class="clearfix special-column-name"  href="https://blog.csdn.net/hexiaolong2009/category_7583191.html" data-report-click='{"mod":"popu_537","spm":"1001.2101.3001.4137","strategy":"pc付费专栏左侧入口","dest":"https://blog.csdn.net/hexiaolong2009/category_7583191.html","ab":"new"}'>
                                      <div class="special-column-bar "></div>
                                      <img src="https://i-blog.csdnimg.cn/columns/default/20201014180756927.png?x-oss-process=image/resize,m_fixed,h_64,w_64" alt="" onerror="this.src='https://i-blog.csdnimg.cn/columns/default/20201014180756922.png?x-oss-process=image/resize,m_fixed,h_64,w_64'">
                                      <span class="">
                                          Android
                                      </span>
                                  </a>
                                  <span class="special-column-num">4篇</span>
                              </li>
                          </ul>
                      </div>
                        <p class="text-center">
                          <a class="flexible-btn-new" data-report-click='{"spm":"3001.10783","strategy":"展开全部"}' data-traigger="true" data-maxheight="0" data-minheight="208px" data-fbox="#aside-content-column" data-flag="flag"><span class="text">展开全部</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-bot-White.png" alt=""></a>
                          <a class="flexible-btn-new-close" data-report-click='{"spm":"3001.10783","strategy":"收起"}'data-traigger="true"  data-minheight="208px" data-fbox="#aside-content-column" data-scroll="true" data-flag="flag"><span class="text">收起</span> <img class="look-more" src="https://csdnimg.cn/release/blogv2/dist/pc/img/arrowup-line-top-White.png" alt=""></a>
                        </p>
                  </div>
          </div>
        </div>
    </aside>
</div>

<div class="recommend-right1  align-items-stretch clearfix" id="rightAsideConcision" data-type="recommend">
    <aside class="recommend-right_aside">
        <div id="recommend-right-concision" >
            <div class='flex-column aside-box groupfile' id="groupfileConcision">
                <div class="groupfile-div1">
                <h3 class="aside-title">目录</h3>
                <div class="align-items-stretch group_item">
                    <div class="pos-box">
                        <div class="scroll-box">
                            <div class="toc-box"></div>
                        </div>
                    </div>
                </div>
                </div>
            </div>
        </div>
    </aside>
</div>

      </div>
      <div class="mask-dark"></div>
        <script type="text/javascript">
        var timert = setInterval(function() {
          sideToolbar = $(".csdn-side-toolbar");
          if (sideToolbar.length > 0) {
            sideToolbar.css('cssText', 'bottom:64px !important;')
            clearInterval(timert);
          }
        }, 200);
        </script>
      <div class="skin-boxshadow"></div>
      <div class="directory-boxshadow"></div>
<div class="comment-side-box-shadow comment-side-tit-close" id="commentSideBoxshadow">
<div class="comment-side-content">
	<div class="comment-side-tit">
		<div class="comment-side-tit-count">评论&nbsp;<span class="count">14</span></div>
	<img class="comment-side-tit-close" src="https://csdnimg.cn/release/blogv2/dist/pc/img/closeBt.png"></div>
  <div id="pcCommentSideBox" class="comment-box comment-box-new2 }" style="display:'block'}">
	
    <div class="comment-edit-box d-flex">
      <div class="user-img">
      </div>
      <form id="commentform">
        <textarea class="comment-content" name="comment_content" id="comment_content" placeholder="欢迎高质量的评论，低质的评论会被折叠" maxlength="1000"></textarea>
        <div class="comment-reward-box" style="background-image: url('https://img-home.csdnimg.cn/images/20230131025301.png');">
          <a class="btn-remove-reward"></a>
          <div class="form-reward-box">
            <div class="info">
              成就一亿技术人!
            </div>
            <div class="price-info">
              拼手气红包<span class="price">6.0元</span>
            </div>
          </div>
        </div>
        <div class="comment-operate-box">
          <div class="comment-operate-l">
            <span id="tip_comment" class="tip">还能输入<em>1000</em>个字符</span>
          </div>
          <div class="comment-operate-c">
            &nbsp;
          </div>
          <div class="comment-operate-r">
            <div class="comment-operate-item comment-reward">
              <img class="comment-operate-img" data-url="https://csdnimg.cn/release/blogv2/dist/pc/img/" src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentReward.png" alt="红包">
              <span class="comment-operate-tip">添加红包</span>
            </div>
            <div class="comment-operate-item comment-emoticon">
              <img class="comment-operate-img" data-url="https://csdnimg.cn/release/blogv2/dist/pc/img/" src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentEmotionIcon.png" alt="表情包">
              <span class="comment-operate-tip">插入表情</span>
              <div class="comment-emoticon-box comment-operate-isshow">
                <div class="comment-emoticon-img-box"></div>
              </div>
            </div>
            <div class="comment-operate-item comment-code">
              <img class="comment-operate-img" data-url="https://csdnimg.cn/release/blogv2/dist/pc/img/" src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentCodeIcon.png" alt="表情包">
              <span class="comment-operate-tip">代码片</span>
              <div class="comment-code-box comment-operate-isshow">
                <ul id="commentCode">
                  <li><a data-code="html">HTML/XML</a></li>
                  <li><a data-code="objc">objective-c</a></li>
                  <li><a data-code="ruby">Ruby</a></li>
                  <li><a data-code="php">PHP</a></li>
                  <li><a data-code="csharp">C</a></li>
                  <li><a data-code="cpp">C++</a></li>
                  <li><a data-code="javascript">JavaScript</a></li>
                  <li><a data-code="python">Python</a></li>
                  <li><a data-code="java">Java</a></li>
                  <li><a data-code="css">CSS</a></li>
                  <li><a data-code="sql">SQL</a></li>
                  <li><a data-code="plain">其它</a></li>
                </ul>
              </div>
            </div>
            <div class="comment-operate-item">
              <input type="hidden" id="comment_replyId" name="comment_replyId">
              <input type="hidden" id="article_id" name="article_id" value="102596791">
              <input type="hidden" id="comment_userId" name="comment_userId" value="">
              <input type="hidden" id="commentId" name="commentId" value="">
              <a data-report-click='{"mod":"1582594662_003","spm":"1001.2101.3001.4227","ab":"new"}'>
              <input type="submit" class="btn-comment btn-comment-input" value="评论">
              </a>
            </div>
          </div>
        </div>
      </form>
    </div>
		<div class="comment-list-container">
			<div class="comment-list-box comment-operate-item">
			</div>
			<div id="lookGoodComment" class="look-good-comment side-look-comment">
				<a class="look-more-comment">查看更多评论<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowDownWhite.png" alt=""></a>
			</div>
			<div id="lookFlodComment" class="look-flod-comment">
					<span class="count"></span>&nbsp;条评论被折叠&nbsp;<a class="look-more-flodcomment">查看</a>
			</div>
			<div class="opt-box text-center">
				<div class="btn btn-sm btn-link-blue" id="btnMoreComment"></div>
			</div>
		</div>
	</div>
	<div id="pcFlodCommentSideBox" class="pc-flodcomment-sidebox">
		<div class="comment-fold-tit"><span id="lookUnFlodComment" class="back"><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowLeftWhite.png" alt=""></span>被折叠的&nbsp;<span class="count"></span>&nbsp;条评论
		 <a href="https://blogdev.blog.csdn.net/article/details/122245662" class="tip" target="_blank">为什么被折叠?</a>
		 <a href="https://bbs.csdn.net/forums/FreeZone" class="park" target="_blank">
		 <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/iconPark.png">到【灌水乐园】发言</a>                                
		</div>
		<div class="comment-fold-content"></div>
		<div id="lookBadComment" class="look-bad-comment side-look-comment">
			<a class="look-more-comment">查看更多评论<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowDownWhite.png" alt=""></a>
		</div>
	</div>
</div>
<div class="comment-rewarddialog-box">
  <div class="form-box">
    <div class="title-box">
      添加红包
      <a class="btn-form-close"></a>
    </div>
    <form id="commentRewardForm">
      <div class="ipt-box">
        <label for="txtName">祝福语</label>
        <div class="ipt-btn-box">
          <input type="text" name="name" id="txtName" autocomplete="off" maxlength="50">
          <a class="btn-ipt btn-random"></a>
        </div>
        <p class="notice">请填写红包祝福语或标题</p>
      </div>
      <div class="ipt-box">
        <label for="txtSendAmount">红包数量</label>
        <div class="ipt-txt-box">
          <input type="text" name="sendAmount" maxlength="4" id="txtSendAmount" placeholder="请填写红包数量(最小10个)" autocomplete="off">
          <span class="after-txt">个</span>
        </div>
        <p class="notice">红包个数最小为10个</p>
      </div>
      <div class="ipt-box">
        <label for="txtMoney">红包总金额</label>
        <div class="ipt-txt-box error">
          <input type="text" name="money" maxlength="5" id="txtMoney" placeholder="请填写总金额(最低5元)" autocomplete="off">
          <span class="after-txt">元</span>
        </div>
        <p class="notice">红包金额最低5元</p>
      </div>
      <div class="balance-info-box">
        <label>余额支付</label>
        <div class="balance-info">
          当前余额<span class="balance">3.43</span>元
          <a href="https://i.csdn.net/#/wallet/balance/recharge" class="link-charge" target="_blank">前往充值 ></a>
        </div>
      </div>
      <div class="opt-box">
        <div class="pay-info">
          需支付：<span class="price">10.00</span>元
        </div>
        <button type="button" class="ml-auto btn-cancel">取消</button>
        <button type="button" class="ml8 btn-submit" disabled="true">确定</button>
      </div>
    </form>
  </div>
</div>
<div class="rr-guide-box">
  <div class="rr-first-box">
    <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/guideRedReward02.png" alt="">
    <button class="btn-guide-known next">下一步</button>
  </div>
  <div class="rr-second-box">
    <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/guideRedReward03.png" alt="">
    <button class="btn-guide-known known">知道了</button>
  </div>
</div>
</div>

<div class="redEnvolope" id="redEnvolope">
  <div class="env-box">
    <div class="env-container">
      <div class="pre-open" id="preOpen">
        <div class="top">
          <header>
            <img class="clearTpaErr" :src="redpacketAuthor.avatar" alt="" />
            <div class="author">成就一亿技术人!</div>
          </header>
          <div class="bot-icon"></div>
        </div>
        <footer>
          <div class="red-openbtn open-start"></div>
          <div class="tip">
            领取后你会自动成为博主和红包主的粉丝
            <a class="rule" target="_blank">规则</a>
          </div>
        </footer>
      </div>
      <div class="opened" id="opened">
        <div class="bot-icon">
          <header>
            <a class="creatorUrl" href="" target="_blank">
              <img class="clearTpaErr" src="https://profile-avatar.csdnimg.cn/default.jpg!2" alt="" />
            </a>
            <div class="author">
              <div class="tt">hope_wisdom</div> 发出的红包
            </div>
          </header>
        </div>
        <div class="receive-box">
          <header></header>
          <div class="receive-list">
          </div>
        </div>
      </div>
    </div>
    <div class="close-btn"></div>
  </div>
</div>
<div id="rewardNew" class="reward-popupbox-new">
	<p class="rewad-title">打赏作者<span class="reward-close"><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/closeBt.png"></span></p>
	<dl class="profile-box">
		<dd>
		<a href="https://blog.csdn.net/hexiaolong2009" data-report-click='{"mod":"popu_379","dest":"https://blog.csdn.net/hexiaolong2009","ab":"new"}'>
			<img src="https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1" class="avatar_pic">
		</a>
		</dd>
		<dt>
			<p class="blog-name">何小龙</p>
			<p class="blog-discript">你的鼓励将是我创作的最大动力</p>
		</dt>
	</dl>
	<div class="reward-box-new">
			<div class="reward-content"><div class="reward-right"></div></div>
	</div>
	<div class="money-box">
    <span class="choose-money choosed" data-id="1">¥1</span>
    <span class="choose-money " data-id="2">¥2</span>
    <span class="choose-money " data-id="4">¥4</span>
    <span class="choose-money " data-id="6">¥6</span>
    <span class="choose-money " data-id="10">¥10</span>
    <span class="choose-money " data-id="20">¥20</span>
	</div>
	<div class="sure-box">
		<div class="sure-box-money">
			<div class="code-box">
				<div class="code-num-box">
					<span class="code-name">扫码支付：</span><span class="code-num">¥1</span>
				</div>
				<div class="code-img-box">
					<div class="renovate">
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/pay-time-out.png">
					<span>获取中</span>
					</div>
				</div>
				<div class="code-pay-box">
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/newWeiXin.png" alt="">
					<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/newZhiFuBao.png" alt="">
					<span>扫码支付</span>
				</div>
			</div>
		</div>
		<div class="sure-box-blance">
			<p class="tip">您的余额不足，请更换扫码支付或<a target="_blank" data-report-click='{"mod":"1597646289_003","spm":"1001.2101.3001.4302"}' href="https://i.csdn.net/#/wallet/balance/recharge?utm_source=RewardVip" class="go-invest">充值</a></p>
			<p class="is-have-money"><a class="reward-sure">打赏作者</a></p>
		</div>
	</div>
</div>
      
      <div class="pay-code">
      <div class="pay-money">实付<span class="pay-money-span" data-nowprice='' data-oldprice=''>元</span></div>
      <div class="content-blance"><a class="blance-bt" href="javascript:;">使用余额支付</a></div>
      <div class="content-code">
        <div id="payCode" data-id="">
          <div class="renovate">
            <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/pay-time-out.png">
            <span>点击重新获取</span>
          </div>
        </div>
        <div class="pay-style"><span><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/weixin.png"></span><span><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/zhifubao.png"></span><span><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/jingdong.png"></span><span class="text">扫码支付</span></div>
      </div>
      <div class="bt-close">
        <svg t="1567152543821" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="10924" xmlns:xlink="http://www.w3.org/1999/xlink" width="12" height="12">
          <defs>
            <style type="text/css"></style>
          </defs>
          <path d="M512 438.378667L806.506667 143.893333a52.032 52.032 0 1 1 73.6 73.621334L585.621333 512l294.485334 294.485333a52.074667 52.074667 0 0 1-73.6 73.642667L512 585.621333 217.514667 880.128a52.053333 52.053333 0 1 1-73.621334-73.642667L438.378667 512 143.893333 217.514667a52.053333 52.053333 0 1 1 73.621334-73.621334L512 438.378667z" fill="" p-id="10925"></path>
        </svg>
      </div>
      <div class="pay-balance">
        <input type="radio" class="pay-code-radio" data-type="details">
        <span class="span">钱包余额</span>
          <span class="balance" style="color:#FC5531;font-size:14px;">0</span>
          <div class="pay-code-tile">
            <img src="https://csdnimg.cn/release/blogv2/dist/pc/img/pay-help.png" alt="">
            <div class="pay-code-content">
              <div class="span">
                <p class="title">抵扣说明：</p>
                <p> 1.余额是钱包充值的虚拟货币，按照1:1的比例进行支付金额的抵扣。<br> 2.余额无法直接购买下载，可以购买VIP、付费专栏及课程。</p>
              </div>
            </div>
          </div>
      </div>
      <a class="pay-balance-con" href="https://i.csdn.net/#/wallet/balance/recharge" target="_blank"><img src="https://csdnimg.cn/release/blogv2/dist/pc/img/recharge.png" alt=""><span>余额充值</span></a>
    </div>
    <div style="display:none;">
      <img src="" onerror='setTimeout(function(){if(!/(csdn.net|iteye.com|baiducontent.com|googleusercontent.com|360webcache.com|sogoucdn.com|bingj.com|baidu.com)$/.test(window.location.hostname)){window.location.href="\x68\x74\x74\x70\x73\x3a\x2f\x2f\x77\x77\x77\x2e\x63\x73\x64\x6e\x2e\x6e\x65\x74"}},3000);'>
    </div>
    <div class="keyword-dec-box" id="keywordDecBox"></div>
  </body>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/axios-83fa28cedf.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_highlight-8defd55d6e.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_common-3c7b273c43.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/components/js/edit_copy_code-a22e5c2c2a.min.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/lib/cboxEditor/1.1.6/embed-editor.min.js" type="text/javascript"></script>
  <link rel="stylesheet" href="https://g.csdnimg.cn/lib/cboxEditor/1.1.6/embed-editor.min.css">
  <link rel="stylesheet" href="https://csdnimg.cn/release/blog_editor_html/release1.6.12/ckeditor/plugins/codesnippet/lib/highlight/styles/atom-one-dark.css">
  <script src="https://g.csdnimg.cn/user-accusation/1.0.6/user-accusation.js" type="text/javascript"></script>
  <script>
    // 全局声明
    if (window.csdn === undefined) {
      window.csdn = {};
    }
    var sideToolbarOpt = {}

    $(function() {
      $(document).on('click', "#toolReportBtnHideNormal,#toolReportBtnHide", function() {
        window.csdn.loginBox.key({
          biz: 'blog',
          subBiz: 'other_service',
          cb: function() {
            window.csdn.feedback({
              "type": 'blog',
              "rtype": 'article',
              "rid": articleId,
              "reportedName": username,
              "submitOptions": {
                "title": articleTitle,
                "contentUrl": articleDetailUrl
              },
              "callback": function() {
                showToast({
                  text: "感谢您的举报，我们会尽快审核！",
                  bottom: '10%',
                  zindex: 9000,
                  speed: 500,
                  time: 1500
                })
              }
            })
          }
        })
      });
    })
      window.csdn.sideToolbar = {
        options: {
          ...sideToolbarOpt,
          theme: 'white',
        }
      }
  </script>
    <script src="https://g.csdnimg.cn/baidu-search/1.0.12/baidu-search.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/download/old_static/js/qrcode.js"></script>
  <script src="https://g.csdnimg.cn/lib/qrcode/1.0.0/qrcode.min.js"></script>
  <script src="https://g.csdnimg.cn/user-ordercart/3.0.1/user-ordercart.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/user-ordertip/5.0.3/user-ordertip.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/order-payment/4.0.5/order-payment.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/pc/js/common-50b21fafc8.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/pc/js/detail-ff634bd1bd.min.js" type="text/javascript"></script>
  <script src="https://csdnimg.cn/release/blogv2/dist/pc/js/column-762ba47480.min.js" type="text/javascript"></script>
    <script src="https://g.csdnimg.cn/side-toolbar/3.6/side-toolbar.js" type="text/javascript"></script>
  <script src="https://g.csdnimg.cn/copyright/1.0.4/copyright.js" type="text/javascript"></script>
  <script>
    $(".MathJax").remove();
    if ($('div.markdown_views pre.prettyprint code.hljs').length > 0) {
      $('div.markdown_views')[0].className = 'markdown_views';
    }
  </script>
  <script type="text/javascript" src="https://csdnimg.cn/release/blog_mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      "HTML-CSS": {
        linebreaks: { automatic: true, width: "94%container" },
        imageFont: null
      },
      tex2jax: {
      preview: "none",
      ignoreClass:"title-article"
      },
      mml2jax: {
      preview: 'none'
      }
    });
  </script>
<script type="text/javascript" crossorigin src="https://g.csdnimg.cn/common/csdn-login-box/csdn-login-box.js"></script></html>
