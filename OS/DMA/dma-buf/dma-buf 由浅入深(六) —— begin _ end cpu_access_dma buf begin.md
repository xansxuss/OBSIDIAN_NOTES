    <!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="utf-8">
    <link rel="canonical" href="https://blog.csdn.net/hexiaolong2009/article/details/102596825"/>
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
    <title>dma-buf 由浅入深（六） —— begin / end cpu_access_dma buf begin-CSDN博客</title>
    <script>
      (function(){ 
        var el = document.createElement("script"); 
        el.src = "https://s3a.pstatp.com/toutiao/push.js?1abfa13dfe74d72d41d83c86d240de427e7cac50c51ead53b2e79d40c7952a23ed7716d05b4a0f683a653eab3e214672511de2457e74e99286eb2c33f4428830"; 
        el.id = "ttzz"; 
        var s = document.getElementsByTagName("script")[0]; 
        s.parentNode.insertBefore(el, s);
      })(window)
    </script>
        <meta name="keywords" content="dma buf begin">
        <meta name="csdn-baidu-search"  content='{"autorun":true,"install":true,"keyword":"dma buf begin"}'>
    <meta name="description" content="文章浏览阅读1.7w次，点赞28次，收藏45次。本篇我们将一起来学习 dma-buf 用于 Cache 同步操作的 begin_cpu_access 和 end_cpu_access 这两个接口。之所以将这两个接口放在第六篇讲解，是因为它们在内核中的使用频率并不高，只有在特殊场景下才派的上用场。_dma buf begin">
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
    <script type="application/ld+json">{"@context":"https://ziyuan.baidu.com/contexts/cambrian.jsonld","@id":"https://blog.csdn.net/hexiaolong2009/article/details/102596825","appid":"1638831770136827","pubDate":"2019-11-26T00:12:46","title":"dma-buf 由浅入深（六） &mdash;&mdash; begin / end cpu_access_dma buf begin-CSDN博客","upDate":"2019-11-26T00:12:46"}</script>
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
        var loginUrl = "http://passport.csdn.net/account/login?from=https://blog.csdn.net/hexiaolong2009/article/details/102596825";
        var blogUrl = "https://blog.csdn.net/";
        var starMapUrl = "https://ai.csdn.net";
        var inscodeHost = "https://inscode.csdn.net";
        var paymentBalanceUrl = "https://csdnimg.cn/release/vip-business-components/vipPaymentBalance.js";
        var appBlogDomain = "https://app-blog.csdn.net";
        var avatar = "https://profile-avatar.csdnimg.cn/13e0f6961a114b5cb09ff65ed930a667_hexiaolong2009.jpg!1";
        var isCJBlog = false;
        var isStarMap = true;
        var articleTitle = "dma-buf 由浅入深（六） —— begin / end cpu_access";
        var articleDesc = "文章浏览阅读1.7w次，点赞28次，收藏45次。本篇我们将一起来学习 dma-buf 用于 Cache 同步操作的 begin_cpu_access 和 end_cpu_access 这两个接口。之所以将这两个接口放在第六篇讲解，是因为它们在内核中的使用频率并不高，只有在特殊场景下才派的上用场。_dma buf begin";
        var articleTitles = "dma-buf 由浅入深（六） —— begin / end cpu_access_dma buf begin-CSDN博客";
        var nickName = "何小龙";
        var articleDetailUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596825";
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
            var toolbarSearchExt = '{\"id\":102596825,\"landingWord\":[\"dma buf begin\"],\"queryWord\":\"\",\"tag\":[\"dma-buf\",\"DRM\",\"内存管理\"],\"title\":\"dma-buf 由浅入深（六） &mdash;&mdash; begin / end cpu_access\"}';
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
      var articleId = 102596825;
        var privateEduData = [];
        var privateData = ["api","翻译","参考资料","内存","cpu"];//高亮数组
      var crytojs = "https://csdnimg.cn/release/blogv2/dist/components/js/crytojs-ca5b8bf6ae.min.js";
      var commentscount = 3;
      var commentAuth = 2;
      var curentUrl = "https://blog.csdn.net/hexiaolong2009/article/details/102596825";
      var myUrl = "https://my.csdn.net/";
      var isGitCodeBlog = false;
      var vipActivityIcon = "https://i-operation.csdnimg.cn/images/df6c67fa661c48eba86beaeb64350df0.gif";
      var isOpenSourceBlog = false;
      var isVipArticle = false;
        var highlight = ["access","begin","由浅入深","内存管理","cpu","buf","dma","end","drm","(",")","六","-"];//高亮数组
        var isRecommendModule = true;
          var isBaiduPre = true;
          var baiduCount = 2;
          var setBaiduJsCount = 10;
        var viewCountFormat = 17990;
      var share_card_url = "https://app-blog.csdn.net/share?article_id=102596825&username=hexiaolong2009"
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
      var baiduKey = "dma buf begin";
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
        var distRequestId = '1766382508835_16991'
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
        var postTime = "2019-11-26 00:12:46"
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
        <h1 class="title-article" id="articleContentId">dma-buf 由浅入深（六） —— begin / end cpu_access</h1>
      </div>
      <div class="article-info-box">
              <div class="up-time">最新推荐文章于&nbsp;2024-06-15 23:08:28&nbsp;发布</div>
          <div class="article-bar-top">
              <div class="bar-content active">
              <span class="article-type-text original">原创</span>
                    <span class="time blog-postTime" data-time="2019-11-26 00:12:46">最新推荐文章于&nbsp;2024-06-15 23:08:28&nbsp;发布</span>
                <span class="border-dian">·</span>
                <span class="read-count">1.7w 阅读</span>
                <div class="read-count-box is-like like-ab-new" data-type="top">
                  <span class="border-dian">·</span>
                  <img class="article-read-img article-heard-img active" style="display:none" id="is-like-imgactive-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png" alt="">
                  <img class="article-read-img article-heard-img" style="display:block" id="is-like-img-new" src="https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png" alt="">
                  <span class="read-count" id="blog-digg-num" style="color:;">
                      28
                  </span>
                </div>
                <span class="border-dian">·</span>
                <a id="blog_detail_zk_collection" class="un-collection" data-report-click='{"mod":"popu_823","spm":"1001.2101.3001.4232","ab":"new"}'>
                  <img class="article-collect-img article-heard-img un-collect-status isdefault" style="display:inline-block" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png" alt="">
                  <img class="article-collect-img article-heard-img collect-status isactive" style="display:none" src="https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png" alt="">
                  <span class="get-collection">
                      45
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
<p>本篇我们将一起来学习 dma-buf 用于 Cache 同步操作的 <em>begin_cpu_access</em> 和 <em>end_cpu_access</em> 这两个接口。之所以将这两个接口放在第六篇讲解&#xff0c;是因为它们在内核中的使用频率并不高&#xff0c;只有在特殊场景下才派的上用场。</p> 
<h3><a id="Cache__14"></a>Cache 一致性</h3> 
<p>下图显示了 CPU 与 DMA 访问 DDR 之间的区别&#xff1a;</p> 
<p><img src="https://i-blog.csdnimg.cn/blog_migrate/c7403fbb7ae1e5c5ce374581709e5391.png#pic_center" alt="在这里插入图片描述" width="400" /></p> 
<p>可以看到&#xff0c;CPU 在访问内存时是要经过 Cache 的&#xff0c;而 DMA 外设则是直接和 DDR 打交道&#xff0c;因此这就存在 Cache 一致性的问题了&#xff0c;<strong>即 Cache 里面的数据是否和 DDR 里面的数据保持一致</strong>。比如 DMA 外设早已将 DDR 中的数据改写了&#xff0c;而 CPU 却浑然不知&#xff0c;仍然在访问 Cache 里面暂存的旧数据。</p> 
<p>所以 Cache 一致性问题&#xff0c;只有在 CPU 参与访问的情况下才会发生。<strong>如果一个 dma-buf 自始自终都只被一个硬件访问&#xff08;要么CPU&#xff0c;要么DMA&#xff09;&#xff0c;那么 Cache 一致性问题就不会存在。</strong></p> 
<blockquote> 
 <p>当然&#xff0c;如果一个 dma-buf 所对应的物理内存本身就是 Uncache 的&#xff08;也叫一致性内存&#xff09;&#xff0c;或者说该 buffer 在被分配时是以 coherent 方式分配的&#xff0c;那么这种情况下&#xff0c;CPU 是不经过 cache 而直接访问 DDR 的&#xff0c;自然 Cache 一致性问题也就不存在了。</p> 
</blockquote> 
<p>有关更多 Cache 一致性的问题&#xff0c;推荐阅读宋宝华的文章<a href="https://blog.csdn.net/juS3Ve/article/details/79135998">《关于DMA ZONE和dma alloc coherent若干误解的彻底澄清》</a>&#xff0c;本文不做赘述。</p> 
<h3><a id="_begin__end__27"></a>为什么需要 begin / end 操作&#xff1f;</h3> 
<p>在前面的<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596772">《dma-buf 由浅入深&#xff08;三&#xff09; —— map attachment》</a>文章中&#xff0c;我们了解到 <code>dma_buf_map_attachment()</code> 函数的一个重要功能&#xff0c;那就是同步 Cache 操作。但是该函数通常使用的是 dma_map_{single,sg} 这种流式 DMA 映射接口来实现 Cache 同步操作&#xff0c;这类接口的特点就是 Cache 同步只是一次性的&#xff0c;即在 dma map 的时候执行一次 Cache Flush 操作&#xff0c;在 dma unmap 的时候执行一次 Cache Invalidate 操作&#xff0c;而这中间的过程是不保证 Cache 和 DDR 上数据的一致性的。因此如果 CPU 在 dma map 和 unmap 之间又去访问了这块内存&#xff0c;那么有可能 CPU 访问到的数据就只是暂存在 Cache 中的旧数据&#xff0c;这就带来了问题。</p> 
<p>那么什么情况下会出现 CPU 在 dma map 和 unmap 期间又去访问这块内存呢&#xff1f;一般不会出现 DMA 硬件正在传输过程中突然 CPU 发起访问的情况&#xff0c;而更多的是在 DMA 硬件发起传输之前&#xff0c;或 DMA 硬件传输完成之后&#xff0c;并且仍然处于 dma map 和 unmap 操作之间的时候&#xff0c;CPU 对这段内存发起了访问。下面举2个例子&#xff1a;</p> 
<ol><li>这是内核文档 <a href="https://elixir.bootlin.com/linux/v4.14.143/source/Documentation/DMA-API-HOWTO.txt#L687" rel="nofollow">DMA-API-HOWTO.txt</a> 中描述的一个网卡驱动例子&#xff0c;非常经典。网卡驱动首先通过 <em>dma_map_single()</em> 将接收缓冲区映射给了网卡 DMA 硬件&#xff0c;此后便发起了 DMA 传输请求&#xff0c;等待网卡接收数据完成。当网卡接收完数据后&#xff0c;会触发中断&#xff0c;此时网卡驱动需要在中断里检查本次传输数据的有效性。如果是有效数据&#xff0c;则调用 dma_unmap_single() 结束本次 DMA 映射&#xff1b;如果不是&#xff0c;则丢弃本次数据&#xff0c;继续等待下一次 DMA 接收的数据。在这个过程中&#xff0c;检查数据有效性是通过 CPU 读取接收缓冲区中的包头来实现的&#xff0c;也只有在数据检查完成后&#xff0c;才能决定是否执行 <em>dma_unmap_single()</em> 操作。因此这里出现了 dma map 和 unmap 期间 CPU 要访问这段内存的需求。</li><li>这是在显示系统中遇到的一个 SPI 屏的例子&#xff0c;也很常见。通常 SPI 屏对总线上传输数据的字节序有严格要求&#xff0c;比如 16bit RGB565 屏幕&#xff0c;要求发送图像数据时&#xff0c;必须先发送高8bit&#xff0c;再发送低8bit。如果平台 SoC SPI 控制器的 DMA 通道只能以byte为单位从低地址向高地址顺序访问&#xff0c;那么它发送出去的数据顺序只能是低8bit在前&#xff0c;高8bit在后&#xff0c;那么就不能满足外设 LCD 的要求&#xff0c;所以需要软件在 SPI 发起传输之前&#xff0c;将显存中的字节序交换一下&#xff0c;此时便涉及到 CPU 访问的需求。也就是说&#xff0c;DRM GEM 驱动首先拿到了 GPU 绘制完成的 buffer&#xff0c;然后对它进行 <em>dma_map_single()</em> 操作&#xff0c;当这块 buffer 交到 CRTC 驱动手里的时候&#xff0c;CPU 需要对该 buffer 再做个字节序交换&#xff0c;然后才送给 SPI DMA&#xff0c;待 DMA 传输完成后执行 <em>dma_unmap_single()</em> 操作。因此这里也出现了 dma map 和 unmap 期间 CPU 要访问这段内存的需求。</li></ol> 
<p>以上第一个例子是 CPU 在 DMA 传输后发起访问&#xff0c;第二个例子是在 DMA 传输前发起访问。针对这种情况&#xff0c;就需要在 CPU 访问内存前&#xff0c;先将 DDR 数据同步到 Cache 中&#xff08;Invalidate&#xff09;&#xff1b;在 CPU 访问结束后&#xff0c;将 Cache 中的数据回写到 DDR 上&#xff08;Flush&#xff09;&#xff0c;以便 DMA 能获取到 CPU 更新后的数据。这也就是 dma-buf 给我们预留 <em>{begin,end}_cpu_access</em> 的原因。</p> 
<h3><a id="Kernel_API_37"></a>Kernel API</h3> 
<p>dma-buf 为我们提供了如下内核 API&#xff0c;用来在 dma map 期间发起 CPU 访问操作&#xff1a;</p> 
<ul><li><code>dma_buf_begin_cpu_access()</code></li><li><code>dma_buf_end_cpu_access()</code></li></ul> 
<p>它们分别对应 <em>dma_buf_ops</em> 中的 <em>begin_cpu_access</em> 和 <em>end_cpu_access</em> 回调接口。</p> 
<p>通常在驱动设计时&#xff0c; <em>begin_cpu_access</em> / <em>end_cpu_access</em> 使用如下流式 DMA 接口来实现 Cache 同步&#xff1a;</p> 
<ul><li><code>dma_sync_single_for_cpu()</code> / <code>dma_sync_single_for_device()</code></li><li><code>dma_sync_sg_for_cpu()</code> / <code>dma_sync_sg_for_device()</code></li></ul> 
<p>CPU 访问内存之前&#xff0c;通过调用 <em>dma_sync_{single,sg}_for_cpu()</em> 来 Invalidate Cache&#xff0c;这样 CPU 在后续访问时才能重新从 DDR 上加载最新的数据到 Cache 上。<br /> CPU 访问内存结束之后&#xff0c;通过调用 <em>dma_sync_{single,sg}_for_device()</em> 来 Flush Cache&#xff0c;将 Cache 中的数据全部回写到 DDR 上&#xff0c;这样后续 DMA 才能访问到正确的有效数据。</p> 
<p>关于更多流式 DMA 映射的介绍&#xff0c;推荐阅读 wowotech 翻译的<a href="http://www.wowotech.net/memory_management/DMA-Mapping-api.html" rel="nofollow">《Dynamic DMA mapping Guide》</a>文章。</p> 
<h3><a id="User_API_53"></a>User API</h3> 
<p>考虑到 <em>mmap()</em> 操作&#xff0c;dma-buf 也为我们提供了 Userspace 的同步接口&#xff0c;通过 <code>DMA_BUF_IOCTL_SYNC</code> <em>ioctl()</em> 来实现。该 cmd 需要一个 <em>struct dma_buf_sync</em> 参数&#xff0c;用于表明当前是 begin 还是 end 操作&#xff0c;是 read 还是 write 操作。</p> 
<blockquote> 
 <p><a href="https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id&#61;c11e391da2a8fe973c3c2398452000bed505851e" rel="nofollow">dma-buf: Add ioctls to allow userspace to flush</a></p> 
</blockquote> 
<p>常用写法如下&#xff1a;</p> 
<pre><code class="prism language-c"><span class="token keyword">struct</span> dma_buf_sync sync <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span> <span class="token number">0</span> <span class="token punctuation">}</span><span class="token punctuation">;</span>

sync<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> DMA_BUF_SYNC_RW <span class="token operator">|</span> DMA_BUF_SYNC_START<span class="token punctuation">;</span>
<span class="token function">ioctl</span><span class="token punctuation">(</span>dmabuf_fd<span class="token punctuation">,</span> DMA_BUF_IOCTL_SYNC<span class="token punctuation">,</span> <span class="token operator">&amp;</span>sync<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token comment">// execute cpu access, for example: memset() ...</span>

sync<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> DMA_BUF_SYNC_RW <span class="token operator">|</span> DMA_BUF_SYNC_END<span class="token punctuation">;</span>
<span class="token function">ioctl</span><span class="token punctuation">(</span>dmabuf_fd<span class="token punctuation">,</span> DMA_BUF_IOCTL_SYNC<span class="token punctuation">,</span> <span class="token operator">&amp;</span>sync<span class="token punctuation">)</span><span class="token punctuation">;</span>

</code></pre> 
<br /> 
<h3><a id="_75"></a>示例</h3> 
<p>本示仅仅用于演示 dma-buf begin / end API 的调用方法&#xff0c;并未考虑真实使用场景的可靠性&#xff0c;各位读者心领神会即可。</p> 
<h4><a id="exporter__78"></a>exporter 驱动</h4> 
<p>我们基于<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791">《dma-buf 由浅入深&#xff08;四&#xff09; —— mmap》</a>中的示例一 exporter-fd.c 文件进行修改&#xff0c;新增 <em>begin_cpu_access</em> 和 <em>end_cpu_access</em> 回调接口&#xff0c;并调用 <em>dma_sync_single_for_{cpu,device}</em> 来完成 Cache 的同步。</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/07/exporter-sync.c">exporter-sync.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/miscdevice.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/slab.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/uaccess.h&gt;</span></span>

<span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf_exported<span class="token punctuation">;</span>
<span class="token function">EXPORT_SYMBOL</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_begin_cpu_access</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span>
				      <span class="token keyword">enum</span> dma_data_direction dir<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	dma_addr_t dma_addr <span class="token operator">&#61;</span> <span class="token function">virt_to_phys</span><span class="token punctuation">(</span>dmabuf<span class="token operator">-&gt;</span>priv<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">dma_sync_single_for_cpu</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> dma_addr<span class="token punctuation">,</span> PAGE_SIZE<span class="token punctuation">,</span> dir<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">exporter_end_cpu_access</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">,</span>
				<span class="token keyword">enum</span> dma_data_direction dir<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	dma_addr_t dma_addr <span class="token operator">&#61;</span> <span class="token function">virt_to_phys</span><span class="token punctuation">(</span>dmabuf<span class="token operator">-&gt;</span>priv<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">dma_sync_single_for_device</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> dma_addr<span class="token punctuation">,</span> PAGE_SIZE<span class="token punctuation">,</span> dir<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">const</span> <span class="token keyword">struct</span> dma_buf_ops exp_dmabuf_ops <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span>
	<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
	<span class="token punctuation">.</span>begin_cpu_access <span class="token operator">&#61;</span> exporter_begin_cpu_access<span class="token punctuation">,</span>
	<span class="token punctuation">.</span>end_cpu_access <span class="token operator">&#61;</span> exporter_end_cpu_access<span class="token punctuation">,</span>
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
	<span class="token keyword">return</span> <span class="token function">copy_to_user</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token keyword">int</span> __user <span class="token operator">*</span><span class="token punctuation">)</span>arg<span class="token punctuation">,</span> <span class="token operator">&amp;</span>fd<span class="token punctuation">,</span> <span class="token keyword">sizeof</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
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

<span class="token function">module_init</span><span class="token punctuation">(</span>exporter_init<span class="token punctuation">)</span><span class="token punctuation">;</span>
</code></pre> 
<h4><a id="importer__166"></a>importer 驱动</h4> 
<p>我们基于<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596761">《dma-buf 由浅入深&#xff08;二&#xff09; —— kmap/vmap》</a>中的 importer-kmap.c 进行修改&#xff0c;如下&#xff1a;</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/07/importer-sync.c">importer-sync.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/module.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/slab.h&gt;</span></span>

<span class="token keyword">extern</span> <span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf_exported<span class="token punctuation">;</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> <span class="token function">importer_test</span><span class="token punctuation">(</span><span class="token keyword">struct</span> dma_buf <span class="token operator">*</span>dmabuf<span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">void</span> <span class="token operator">*</span>vaddr<span class="token punctuation">;</span>

	<span class="token function">dma_buf_begin_cpu_access</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> DMA_FROM_DEVICE<span class="token punctuation">)</span><span class="token punctuation">;</span>

	vaddr <span class="token operator">&#61;</span> <span class="token function">dma_buf_kmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;read from dmabuf kmap: %s\n&#34;</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token keyword">char</span> <span class="token operator">*</span><span class="token punctuation">)</span>vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">dma_buf_kunmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>

	vaddr <span class="token operator">&#61;</span> <span class="token function">dma_buf_vmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">pr_info</span><span class="token punctuation">(</span><span class="token string">&#34;read from dmabuf vmap: %s\n&#34;</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token keyword">char</span> <span class="token operator">*</span><span class="token punctuation">)</span>vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">dma_buf_vunmap</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> vaddr<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">dma_buf_end_cpu_access</span><span class="token punctuation">(</span>dmabuf<span class="token punctuation">,</span> DMA_FROM_DEVICE<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">static</span> <span class="token keyword">int</span> __init <span class="token function">importer_init</span><span class="token punctuation">(</span><span class="token keyword">void</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">return</span> <span class="token function">importer_test</span><span class="token punctuation">(</span>dmabuf_exported<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token function">module_init</span><span class="token punctuation">(</span>importer_init<span class="token punctuation">)</span><span class="token punctuation">;</span>
</code></pre> 
<p>该 importer 驱动将原来的 kmap / vmap 操作放到了 begin / end 操作中间&#xff0c;以确保读取数据的正确性&#xff08;虽然在本示例中没有任何意义&#xff09;。</p> 
<h4><a id="userspace__205"></a>userspace 程序</h4> 
<p>我们基于<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596791">《dma-buf 由浅入深&#xff08;四&#xff09; —— mmap》</a>中的示例一 mmap_dmabuf.c 文件进行修改&#xff0c;如下&#xff1a;</p> 
<p><a href="https://github.com/hexiaolong2008/sample-code/blob/master/dma-buf/07/dmabuf-test/dmabuf_sync.c">dmabuf_sync.c</a></p> 
<pre><code class="prism language-c"><span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;string.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;stdio.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;stdlib.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;errno.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;fcntl.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;unistd.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;sys/types.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;sys/stat.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;sys/mman.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;sys/ioctl.h&gt;</span></span>
<span class="token macro property">#<span class="token directive keyword">include</span> <span class="token string">&lt;linux/dma-buf.h&gt;</span></span>

<span class="token keyword">int</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token keyword">int</span> argc<span class="token punctuation">,</span> <span class="token keyword">char</span> <span class="token operator">*</span>argv<span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token punctuation">{<!-- --></span>
	<span class="token keyword">int</span> fd<span class="token punctuation">;</span>
	<span class="token keyword">int</span> dmabuf_fd <span class="token operator">&#61;</span> <span class="token number">0</span><span class="token punctuation">;</span>
	<span class="token keyword">struct</span> dma_buf_sync sync <span class="token operator">&#61;</span> <span class="token punctuation">{<!-- --></span> <span class="token number">0</span> <span class="token punctuation">}</span><span class="token punctuation">;</span>

	fd <span class="token operator">&#61;</span> <span class="token function">open</span><span class="token punctuation">(</span><span class="token string">&#34;/dev/exporter&#34;</span><span class="token punctuation">,</span> O_RDONLY<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token function">ioctl</span><span class="token punctuation">(</span>fd<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token operator">&amp;</span>dmabuf_fd<span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">close</span><span class="token punctuation">(</span>fd<span class="token punctuation">)</span><span class="token punctuation">;</span>

	sync<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> DMA_BUF_SYNC_READ <span class="token operator">|</span> DMA_BUF_SYNC_START<span class="token punctuation">;</span>
	<span class="token function">ioctl</span><span class="token punctuation">(</span>dmabuf_fd<span class="token punctuation">,</span> DMA_BUF_IOCTL_SYNC<span class="token punctuation">,</span> <span class="token operator">&amp;</span>sync<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">char</span> <span class="token operator">*</span>str <span class="token operator">&#61;</span> <span class="token function">mmap</span><span class="token punctuation">(</span><span class="token constant">NULL</span><span class="token punctuation">,</span> <span class="token number">4096</span><span class="token punctuation">,</span> PROT_READ<span class="token punctuation">,</span> MAP_SHARED<span class="token punctuation">,</span> dmabuf_fd<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
	<span class="token function">printf</span><span class="token punctuation">(</span><span class="token string">&#34;read from dmabuf mmap: %s\n&#34;</span><span class="token punctuation">,</span> str<span class="token punctuation">)</span><span class="token punctuation">;</span>

	sync<span class="token punctuation">.</span>flags <span class="token operator">&#61;</span> DMA_BUF_SYNC_READ <span class="token operator">|</span> DMA_BUF_SYNC_END<span class="token punctuation">;</span>
	<span class="token function">ioctl</span><span class="token punctuation">(</span>dmabuf_fd<span class="token punctuation">,</span> DMA_BUF_IOCTL_SYNC<span class="token punctuation">,</span> <span class="token operator">&amp;</span>sync<span class="token punctuation">)</span><span class="token punctuation">;</span>

	<span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
</code></pre> 
<p>该测试程序将原来的 <em>mmap()</em> 操作放到了 <em>ioctl</em> SYNC_START / SYNC_END 之间&#xff0c;以确保读取数据的正确性&#xff08;虽然在本示例中没有任何意义&#xff09;。</p> 
<h3><a id="_247"></a>开发环境</h3> 
<table><thead><tr><th align="left"></th><th align="left"></th></tr></thead><tbody><tr><td align="left">内核源码</td><td align="left"><a href="https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x/linux-4.14.143.tar.xz" rel="nofollow">4.14.143</a></td></tr><tr><td align="left">示例源码</td><td align="left"><a href="https://github.com/hexiaolong2008/sample-code/tree/master/dma-buf/07">hexiaolong2008-GitHub/sample-code/dma-buf/07</a></td></tr><tr><td align="left">开发平台</td><td align="left">Ubuntu14.04/16.04</td></tr><tr><td align="left">运行平台</td><td align="left"><a href="https://github.com/hexiaolong2008/my-qemu">my-qemu 仿真环境</a></td></tr></tbody></table>
<h3><a id="_255"></a>运行</h3> 
<p>在 my-qemu 仿真环境中执行如下命令&#xff1a;</p> 
<pre><code class="prism language-handlebars"><span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/exporter-sync.ko</span>
<span class="token punctuation">#</span> <span class="token variable">insmod</span> <span class="token punctuation">/</span><span class="token variable">lib</span><span class="token punctuation">/</span><span class="token variable">modules</span><span class="token punctuation">/</span><span class="token number">4.14</span><span class="token number">.143</span><span class="token block keyword">/kernel/drivers/dma-buf/importer-sync.ko</span>
</code></pre> 
<p>输出结果如下&#xff1a;</p> 
<pre><code>read from dmabuf kmap: hello world!
read from dmabuf vmap: hello world!
</code></pre> 
<p>接着执行如下命令&#xff1a;</p> 
<pre><code class="prism language-handlebars"><span class="token punctuation">#</span> <span class="token punctuation">.</span><span class="token punctuation">/</span><span class="token variable">dmabuf_sync</span>
</code></pre> 
<p>输出&#xff1a;</p> 
<pre><code>read from dmabuf mmap: hello world!
</code></pre> 
<p>输出的结果其实和之前的程序没有任何区别。</p> 
<h3><a id="_278"></a>总结</h3> 
<ol><li>只有在 DMA map/unmap 期间 CPU 又要访问内存的时候&#xff0c;才有必要使用 begin / end 操作&#xff1b;</li><li><em>{ begin,end }_cpu_access</em> 实际是 <em>dma_sync</em>()* 接口的封装&#xff0c;目的是要 invalidate 或 flush cache&#xff1b;</li><li>Usespace 通过 <em>DMA_BUF_IOCTL_SYNC</em> 来触发 begin / end 操作&#xff1b;</li></ol> 
<h3><a id="_283"></a>参考资料</h3> 
<ol><li>wowotech 翻译&#xff1a;<a href="http://www.wowotech.net/memory_management/DMA-Mapping-api.html" rel="nofollow">Dynamic DMA mapping Guide</a></li><li>user space: <a href="https://elixir.bootlin.com/linux/v5.0/source/tools/testing/selftests/android/ion/ionmap_test.c" rel="nofollow">linux-5.0/tools/testing/selftests/android/ion/ionmap_test.c</a></li><li>kernel space: <a href="https://elixir.bootlin.com/linux/v4.14.143/source/drivers/gpu/drm/tinydrm/mipi-dbi.c#L156" rel="nofollow">linux-4.14/drivers/gpu/drm/tinydrm/mipi-dbi.c</a></li></ol> 
<br /> 
<br /> 
<br /> 
<p>上一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802">《dma-buf 由浅入深&#xff08;五&#xff09;—— File》</a><br /> 下一篇&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/102596845">《dma-buf 由浅入深&#xff08;七&#xff09; —— alloc page 版本》</a><br /> 文章汇总&#xff1a;<a href="https://blog.csdn.net/hexiaolong2009/article/details/83720940">《DRM&#xff08;Direct Rendering Manager&#xff09;学习简介》</a></p>
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
                    28
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
                <span class="count get-collection " data-num="45" id="get-collection">
                    45
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
                    3
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
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/MHD0815/article/details/152240329"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~YuanLiJiHua~PaidSort-1-152240329-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~YuanLiJiHua~PaidSort","dest":"https://blog.csdn.net/MHD0815/article/details/152240329"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/MHD0815/article/details/152240329" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~YuanLiJiHua~PaidSort-1-152240329-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~YuanLiJiHua~PaidSort","dest":"https://blog.csdn.net/MHD0815/article/details/152240329"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPaidSort-1-152240329-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPaidSort-1-152240329-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">基于Linux的PCIe设备驱动开发入门到精通<em>-</em><em>-</em>6.2 Linux内核<em>DMA</em> API：<em>dma</em>_map_single<em>(</em><em>)</em>, <em>dma</em>_unmap_single<em>(</em><em>)</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/MHD0815" target="_blank"><span class="blog-title">MHD0815的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">10-07</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					107
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/MHD0815/article/details/152240329" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6661.1","mod":"popu_871","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant_t0.none-task-blog-2~default~YuanLiJiHua~PaidSort-1-152240329-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~YuanLiJiHua~PaidSort","dest":"https://blog.csdn.net/MHD0815/article/details/152240329"}'  data-report-query='spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPaidSort-1-152240329-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPaidSort-1-152240329-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">本文深入解析了Linux内核<em>DMA</em> API中的核心函数<em>dma</em>_map_single<em>(</em><em>)</em>和<em>dma</em>_unmap_single<em>(</em><em>)</em>。文章首先指出直接使用物理地址进行<em>DMA</em>传输的危险性，强调必须使用内核API来确保IOMMU兼容性、缓存一致性和平台无关性。详细讲解了这两个API的参数、返回值和使用场景，特别强调了direction参数对缓存操作的决定性影响。通过典型<em>DMA</em>传输流程示例，展示了map<em>-</em>unmap的配对使用模式。最后总结了关键注意事项，包括正确设置方向、及时取消映射以及错误处理等。文章预告下一讲将介绍一</div>
			</a>
		</div>
	</div>
</div>
                </div>
            <script src="https://csdnimg.cn/release/blogv2/dist/components/js/pc_wap_commontools-829a4838ae.min.js" type="text/javascript" async></script>
              <div class="second-recommend-box recommend-box ">
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/IT_Beijing_BIT/article/details/140393205"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-140393205-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140393205"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/IT_Beijing_BIT/article/details/140393205" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-140393205-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140393205"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140393205-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140393205-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">缓冲区共享和同步<em>dma</em>_<em>buf</em> 之二</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/IT_Beijing_BIT" target="_blank"><span class="blog-title">IT_Beijing_BIT的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">07-13</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1695
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/IT_Beijing_BIT/article/details/140393205" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.1","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~PaidSort-1-140393205-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"1","strategy":"2~default~BlogCommendFromBaidu~PaidSort","dest":"https://blog.csdn.net/IT_Beijing_BIT/article/details/140393205"}'  data-report-query='spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140393205-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-140393205-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">可以使用作为一个sync_file的<em>dma</em><em>-</em><em>buf</em> 文件描述符，  执行 <em>DMA</em>_<em>BUF</em>_IOCTL_EXPORT_SYNC_FILE，以获得当前围栏集。在访问映射之前，客户端必须使用 <em>DMA</em>_<em>BUF</em>_SYNC_START 和适当的读/写标志调用 <em>DMA</em>_<em>BUF</em>_IOCTL_SYNC。为实现与其他 <em>dma</em><em>-</em><em>buf</em> 使用者的隐式同步，用户空间可以执行 <em>DMA</em>_<em>BUF</em>_IOCTL_IMPORT_SYNC_FILE 将sync_file 插入到 <em>dma</em><em>-</em><em>buf</em> 中。将sync_file插入到<em>dma</em><em>-</em><em>buf</em>中。</div>
			</a>
		</div>
	</div>
</div>
              </div>
<a id="commentBox" name="commentBox"></a>
  <div id="pcCommentBox" class="comment-box comment-box-new2 unlogin-comment-box-new" style="display:none">
      <div class="unlogin-comment-model">
          <span class="unlogin-comment-tit">3&nbsp;条评论</span>
        <span class="unlogin-comment-text">您还未登录，请先</span>
        <span class="unlogin-comment-bt">登录</span>
        <span class="unlogin-comment-text">后发表或查看评论</span>
      </div>
  </div>
  <div class="blog-comment-box-new" style="display: none;">
        <h1>3 条评论</h1>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/yuhaojin">
              <img src="https://profile-avatar.csdnimg.cn/673ff47616774da09243a519d2b3b3fe_yuhaojin.jpg!1"
                alt="yuhaojin" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/yuhaojin">
                      <span class="name ">yuhaojin</span></a>
                    <span class="date" title="2021-07-23 14:55:53">2021.07.23</span>
                    <div class="new-comment">您好，请问下&ldquo;dma_buf_end_cpu_access(dmabuf, DMA_FROM_DEVICE);&rdquo;里面的DMA_FROM_DEVICE是不是应该改为DMA_TO_DEVICE？</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/zsj06130675">
              <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                alt="zsj06130675" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/zsj06130675">
                      <span class="name ">zsj06130675</span></a>
                    <span class="date" title="2019-12-13 16:39:08">2019.12.13</span>
                    <div class="new-comment">君子好文！</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
      <ul>
          <li>
            <a target="_blank" href="https://blog.csdn.net/billykun">
              <img src="https://profile-avatar.csdnimg.cn/default.jpg!1"
                alt="billykun" class="avatar">
            </a>
            <div class="right-box">
              <div class="new-info-box clearfix">
                <div class="comment-top">
                  <div class="user-box">
                    <a class="name-href" target="_blank"  href="https://blog.csdn.net/billykun">
                      <span class="name ">billykun</span></a>
                    <span class="date" title="2019-12-10 11:18:15">2019.12.10</span>
                    <div class="new-comment">学习了这系列的文章，文章写的真好，大赞！</div>
                  </div>
                </div>
              </div>
            </div>
          </li>
      </ul>
    </div>
              <div class="recommend-box insert-baidu-box recommend-box-style ">
                <div class="recommend-item-box no-index" style="display:none"></div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/weixin_71478434/article/details/126559562"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-2-126559562-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_71478434/article/details/126559562"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/weixin_71478434/article/details/126559562" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-2-126559562-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_71478434/article/details/126559562"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-126559562-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-126559562-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">看完秒懂：Linux <em>DMA</em> mapping机制分析</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_71478434" target="_blank"><span class="blog-title">weixin_71478434的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">08-27</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					5882
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/weixin_71478434/article/details/126559562" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.2","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-2-126559562-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"2","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/weixin_71478434/article/details/126559562"}'  data-report-query='spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-126559562-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-126559562-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">Linux内核中提供了两种<em>dma</em> mapping的接口：Consistent mapping和Stream mapping。通常在使用consistent <em>dma</em> mapping时，首先需要通过。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/anyegongjuezjd/article/details/136271103"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-3-136271103-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/anyegongjuezjd/article/details/136271103"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/anyegongjuezjd/article/details/136271103" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-3-136271103-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/anyegongjuezjd/article/details/136271103"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-136271103-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-136271103-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">浅析Linux设备<em>DMA</em>机制：<em>DMA</em>内存映射</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/anyegongjuezjd" target="_blank"><span class="blog-title">Aspiresky的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">02-25</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2311
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/anyegongjuezjd/article/details/136271103" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.3","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-3-136271103-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"3","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/anyegongjuezjd/article/details/136271103"}'  data-report-query='spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-136271103-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-136271103-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">现代计算机系统中，<em>CPU</em>访问内存需要经过Cache，但外部设备通常不感知Cache的存在，因此<em>CPU</em>和外设在访问<em>DMA</em>内存时，必须谨慎处理内存数据的一致性问题。为了处理这种一致性问题，同时为了兼顾多种设备类型，Linux系统会采用不同的规则来映射<em>DMA</em>内存，开发者遵循这套规则对<em>DMA</em>内存进行操作。</div>
			</a>
		</div>
	</div>
</div>
		<dl id="recommend-item-box-tow" class="recommend-item-box type_blog clearfix">
			
		</dl>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/wmzjzwlzs/article/details/125229340"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-125229340-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/wmzjzwlzs/article/details/125229340"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/wmzjzwlzs/article/details/125229340" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-125229340-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/wmzjzwlzs/article/details/125229340"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-125229340-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-125229340-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">ion_<em>dma</em>_<em>buf</em>_<em>begin</em>_<em>cpu</em>_<em>access</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/wmzjzwlzs" target="_blank"><span class="blog-title">wmzjzwlzs的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">06-10</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					341
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/wmzjzwlzs/article/details/125229340" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.4","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-125229340-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"4","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/wmzjzwlzs/article/details/125229340"}'  data-report-query='spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-125229340-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-125229340-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">static int ion_<em>dma</em>_<em>buf</em>_<em>begin</em>_<em>cpu</em>_<em>access</em><em>(</em>struct <em>dma</em>_<em>buf</em> *<em>dma</em><em>buf</em>,
                    enum <em>dma</em>_data_direction direction<em>)</em>
{
    struct ion_<em>buf</em>fer *<em>buf</em>fer = <em>dma</em><em>buf</em><em>-</em>&gt;priv;
    void *vaddr;
    struct ion_<em>dma</em>_<em>buf</em>_attachment *a;
    int ret = 0;    /*
     * TODO</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/7940330"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-7940330-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/7940330" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-7940330-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-7940330-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-7940330-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
			<a href="https://devpress.csdn.net/v1/article/detail/7940330" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.5","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-5-7940330-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"5","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/7940330"}'  data-report-query='spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-7940330-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-5-7940330-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/relax33/article/details/128319124"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-6-128319124-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/relax33/article/details/128319124" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-6-128319124-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
			<a href="https://blog.csdn.net/relax33/article/details/128319124" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.6","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-6-128319124-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"6","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/relax33/article/details/128319124"}'  data-report-query='spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-6-128319124-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">ION <em>DMA</em><em>-</em><em>BUF</em> IOMMU</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/yuuu_cheer/article/details/129549854"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-7-129549854-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/yuuu_cheer/article/details/129549854"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/yuuu_cheer/article/details/129549854" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-7-129549854-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/yuuu_cheer/article/details/129549854"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-129549854-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-129549854-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>DRM</em>驱动代码分析：gem_prime_import</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/yuuu_cheer" target="_blank"><span class="blog-title">yuuu_cheer的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">04-28</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					886
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/yuuu_cheer/article/details/129549854" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.7","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~OPENSEARCH~Rate-7-129549854-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"7","strategy":"2~default~OPENSEARCH~Rate","dest":"https://blog.csdn.net/yuuu_cheer/article/details/129549854"}'  data-report-query='spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-129549854-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-7-129549854-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>DRM</em>驱动代码分析：gem_prime_import</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_wk_aigc_column clearfix" data-url="https://wenku.csdn.net/column/5k5wtimmja"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/5k5wtimmja"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/column/5k5wtimmja" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/5k5wtimmja"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">ESP32驱动TFT屏幕卡顿真相：如何定位<em>CPU</em>与<em>DMA</em>资源竞争（90%开发者忽略的关键点）</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block"></span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/column/5k5wtimmja" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.8","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/5k5wtimmja"}'  data-report-query='spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"># 1. ESP32驱动TFT屏幕的常见卡顿现象解析 ...根本原因并非处理器性能不足，而是**<em>CPU</em>与<em>DMA</em>协同不当**、SPI总线带宽饱和以及任务优先级配置不合理所致。  后续章节将从ESP32系统架构出发，逐步解析资源竞争机</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_wk_aigc_column clearfix" data-url="https://wenku.csdn.net/column/238sp25xzi"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/238sp25xzi"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/column/238sp25xzi" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/238sp25xzi"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">RoCEv2 vs TCP_IP性能大比拼：FPGA平台实测数据深度解析</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block"></span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/column/238sp25xzi" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.9","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/238sp25xzi"}'  data-report-query='spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">!... # 摘要  本文围绕RoCEv2与TCP/IP协议展开系统性对比研究，重点分析两者在协议架构、传输机制及拥塞控制等方面的异同。结合FPGA平台的硬件加速特性，探讨其在高性能网络中的实现方式与性能优势。...</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_wk_aigc_column clearfix" data-url="https://wenku.csdn.net/column/2mp72dx36j"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/2mp72dx36j"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/column/2mp72dx36j" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/2mp72dx36j"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">【RT<em>-</em>Thread系统移植全栈指南】：掌握15个核心步骤，从零实现嵌入式OS底层适配</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block"></span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/column/2mp72dx36j" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.10","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/column/2mp72dx36j"}'  data-report-query='spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">!...# 1. RT<em>-</em>Thread系统移植概述与核心概念  在嵌入式实时操作系统（RTOS）领域，RT<em>-</em>Thread因其模块化设计、良好的可裁剪性与丰富的中间件生态，广泛应用于工业控制、物联网终端等高性能场景。系统移植是将RT<em>-</em>Threa</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596772"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-11-102596772-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596772" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-11-102596772-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596772-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596772-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
			<a href="https://devpress.csdn.net/v1/article/detail/102596772" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.11","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-11-102596772-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"11","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596772"}'  data-report-query='spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596772-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-11-102596772-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在上一篇《kmap/vmap》中，我们学习了如何使用 <em>CPU</em> 在 kernel 空间访问 <em>dma</em><em>-</em><em>buf</em> 物理内存，但如果使用<em>CPU</em>直接去访问 memory，那么性能会大大降低。因此，<em>dma</em><em>-</em><em>buf</em> 在内核中出现频率最高的还是它的 <em>dma</em>_<em>buf</em>_attach<em>(</em><em>)</em> 和 <em>dma</em>_<em>buf</em>_map_attachment<em>(</em><em>)</em> 接口。本篇我们就一起来学习如何通过这两个 API 来实现 <em>DMA</em> 硬件对 <em>dma</em><em>-</em><em>buf</em> 物理内存的访问。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596744"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596744-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596744" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596744-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（一） &mdash;&mdash; 最简单的 <em>dma</em><em>-</em><em>buf</em> 驱动程序</div>
					<div class="tag">热门推荐</div>
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
			<a href="https://devpress.csdn.net/v1/article/detail/102596744" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.12","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-12-102596744-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"12","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596744"}'  data-report-query='spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-102596744-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">如果你和我一样，是一位从事Android多媒体底层开发的工程师，那么你对 <em>dma</em><em>-</em><em>buf</em> 这个词语一定不会陌生，因为不管是 Video、Camera 还是 Display、GPU，它们的<em>buf</em>fer都来自于ION，而 ION 正是基于 <em>dma</em><em>-</em><em>buf</em> 实现的。

假如你对 <em>dma</em><em>-</em><em>buf</em> 的理解并不深刻，又期望找个时间来彻底公关一下，那么很高兴，这几篇文章一定能让你对 <em>dma</em><em>-</em><em>buf</em> 有个更深入、更透彻的理解。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/abc3240660/article/details/81942190"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-81942190-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/abc3240660/article/details/81942190" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-81942190-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-81942190-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-81942190-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
			<a href="https://blog.csdn.net/abc3240660/article/details/81942190" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.13","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-13-81942190-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"13","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/abc3240660/article/details/81942190"}'  data-report-query='spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-81942190-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-13-81942190-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
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
<div class="recommend-item-box type_blog clearfix" data-url="https://killerp.blog.csdn.net/article/details/139710556"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-139710556-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://killerp.blog.csdn.net/article/details/139710556"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://killerp.blog.csdn.net/article/details/139710556" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-139710556-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://killerp.blog.csdn.net/article/details/139710556"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-139710556-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-139710556-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux <em>DMA</em><em>-</em><em>Buf</em>驱动框架</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/weixin_44821644" target="_blank"><span class="blog-title">杀手的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">06-15</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					4146
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://killerp.blog.csdn.net/article/details/139710556" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.14","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-14-139710556-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"14","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://killerp.blog.csdn.net/article/details/139710556"}'  data-report-query='spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-139710556-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-14-139710556-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>dma</em><em>buf</em> 是一个驱动间共享<em>buf</em> 的机制，他的简单使用场景如下：用户从<em>DRM</em>（显示驱动）申请一个<em>dma</em><em>buf</em>，把<em>dma</em><em>buf</em> 设置给GPU驱动，并启动GPU将数据输出到<em>dma</em><em>buf</em>，GPU输出完成后，再将<em>dma</em><em>buf</em>设置到<em>DRM</em> 驱动，完成画面的显示。在这个过程中通过共享<em>dma</em><em>buf</em>的方式，避免了GPU输出数据拷贝到<em>drm</em> frame <em>buf</em>f的动作。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/fs3296/article/details/125387687"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-125387687-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/fs3296/article/details/125387687"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/fs3296/article/details/125387687" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-125387687-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/fs3296/article/details/125387687"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-125387687-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-125387687-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux图形子系统之<em>dma</em><em>-</em><em>buf</em></div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/fs3296" target="_blank"><span class="blog-title">fs3296的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">06-21</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2997
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/fs3296/article/details/125387687" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.15","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-15-125387687-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"15","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/fs3296/article/details/125387687"}'  data-report-query='spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-125387687-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-15-125387687-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em>是linux内核提供的一种机制，用于不同模块实现内存共享。它提供生产者和消费者模式来实现不同模块对内存共享同时，不用关心各个模块的内部实现细节，从而解耦。在<em>drm</em>框架中也集成了<em>dma</em><em>-</em><em>buf</em>方式的<em>内存管理</em>。<em>drm</em>通过<em>DRM</em>_IOCTL_PRIME_HANDLE_TO_FD实现将一个gem对象句柄转为<em>dma</em><em>-</em><em>buf</em>的fd。其中会调用struct <em>drm</em>_driver的prime_handle_to_fd回调，<em>drm</em>_gem_prime_handle_to_fd函数是prime_handle_to</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/junwua/article/details/125197539"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-125197539-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/junwua/article/details/125197539"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/junwua/article/details/125197539" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-125197539-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/junwua/article/details/125197539"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-125197539-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-125197539-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">linux<em>内存管理</em>机制<em>-</em><em>-</em>学习整理汇总 <em>dma</em><em>-</em><em>buf</em>（3）</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/junwua" target="_blank"><span class="blog-title">求变，从思想开始</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">06-09</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					5693
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/junwua/article/details/125197539" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.16","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-16-125197539-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"16","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/junwua/article/details/125197539"}'  data-report-query='spm=1001.2101.3001.6650.16&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-125197539-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-16-125197539-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">linux <em>内存管理</em>和使用 <em>dma</em>_<em>buf</em></div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://devpress.csdn.net/v1/article/detail/102596802"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-17-102596802-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596802"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://devpress.csdn.net/v1/article/detail/102596802" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-17-102596802-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596802"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-17-102596802-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-17-102596802-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（五） &mdash;&mdash; File</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/hexiaolong2009" target="_blank"><span class="blog-title">hexiaolong2009的专栏</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">11-26</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					2万+
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://devpress.csdn.net/v1/article/detail/102596802" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.17","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-17-102596802-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"17","strategy":"2~default~BlogCommendFromBaidu~activity","dest":"https://devpress.csdn.net/v1/article/detail/102596802"}'  data-report-query='spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-17-102596802-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-17-102596802-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">在上一篇《<em>dma</em><em>-</em><em>buf</em> <em>由浅入深</em>（四）&mdash;&mdash; mmap》中，曾提到过 <em>dma</em>_<em>buf</em>_fd<em>(</em><em>)</em> 这个函数，该函数用于创建一个新的 fd，并与 <em>dma</em><em>-</em><em>buf</em> 的文件关联起来。本篇我们一起来重点学习 <em>dma</em><em>-</em><em>buf</em> 与 file 相关的操作接口，以及它们的注意事项。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_blog clearfix" data-url="https://blog.csdn.net/mk_archermind/article/details/137616125"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-137616125-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/mk_archermind/article/details/137616125"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://blog.csdn.net/mk_archermind/article/details/137616125" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-137616125-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/mk_archermind/article/details/137616125"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-137616125-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-137616125-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">Linux <em>dma</em><em>-</em><em>buf</em> 梳理记录</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info">
					<a href="https://blog.csdn.net/mk_archermind" target="_blank"><span class="blog-title">mk_archermind的博客</span></a>
				</div>
				<div class="info display-flex">
					<span class="info-block time">04-10</span>
					<span class="info-block read"><img class="read-img" src="https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png" alt="">
					1543
					</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://blog.csdn.net/mk_archermind/article/details/137616125" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.18","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-18-137616125-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"18","strategy":"2~default~BlogCommendFromBaidu~Rate","dest":"https://blog.csdn.net/mk_archermind/article/details/137616125"}'  data-report-query='spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-137616125-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-18-137616125-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1"><em>dma</em><em>-</em>heap.c: 提供heap的操作接口，注册heap 和 对外heap的操作接口如ioctl/open，主要是字符设备。<em>dma</em><em>-</em><em>buf</em>.c: 提供基础函数，主要是<em>dma</em>内存操作函数，包括mmap，munmap，attach,detach等。一直想学习，好久不看Linux代码，看不太明白了~费了大力气，梳理了一些东西，权当记录。system heap是可以物理不连续的，内存分配是通过 alloc_pages。cma是物理连续的，内存分配是通过 cma_alloc。</div>
			</a>
		</div>
	</div>
</div>
<div class="recommend-item-box type_chatgpt clearfix" data-url="https://wenku.csdn.net/answer/6m8pdk8q94"  data-report-view='{"ab":"new","spm":"1001.2101.3001.6650.19","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/6m8pdk8q94"}'>
	<div class="content-box">
		<div class="content-blog display-flex">
			<div class="title-box">
				<a href="https://wenku.csdn.net/answer/6m8pdk8q94" class="tit" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.19","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/6m8pdk8q94"}'  data-report-query='spm=1001.2101.3001.6650.19&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
					<div class="left ellipsis-online ellipsis-online-1">module SCCB_CTRL
<em>(</em>
    input       clk       , //100M
    input       rst_n     ,
    output      scl       , 
    output      XCLK      ,
    inout       sda       
    // input       initial_start ,
    // output      cam_rst,            //摄像头复位信号 <em>(</em>低电平有效<em>)</em>
    // output      cam_pwdn ,          //摄像头功耗控制 <em>(</em>0:正常工作, 1:休眠模式<em>)</em>
    // output      work_done   //SCCB读写完成信号
<em>)</em>;

//100M
`define SCL_POSEDGE <em>(</em>cnt == 16&#39;d0<em>)</em>
`define SCL_NEGEDGE <em>(</em>cnt == 16&#39;d160<em>)</em>
`define SCL_HIG_MID <em>(</em>cnt == 16&#39;d80<em>)</em>
`define SCL_LOW_MID <em>(</em>cnt == 16&#39;d280<em>)</em>
//50M
// `define SCL_POSEDGE <em>(</em>cnt == 7&#39;d0<em>)</em>
// `define SCL_NEGEDGE <em>(</em>cnt == 7&#39;d50<em>)</em>
// `define SCL_HIG_MID <em>(</em>cnt == 7&#39;d25<em>)</em>
// `define SCL_LOW_MID <em>(</em>cnt == 7&#39;d87<em>)</em>


//时钟频率、寄存器地址的声明
parameter    CLK_FRQ       = 100_000_000 ;
parameter    CNT_3SEC=CLK_FRQ*3<em>-</em>1       ;
parameter    CNT_1SEC=CLK_FRQ<em>-</em>1         ;
parameter   slave_addr     = 7&#39;b0111_100;

// assign cam_rst = 1;
// assign cam_pwdn = 0;
// wire
wire initial_done;
wire work_start;
wire [23:0] initial_data;
wire write_start_flag_wire; //使能信号
wire sda_in;            //sda输入寄存器

//reg
reg    [7:0]   data_tx         ;  //写入数据寄存器
reg    [7:0]   data_rx         ;  //读出数据寄存器S
reg    [7:0]   w_slave_addr_<em>buf</em>; //从设备地址寄存器（地址存高7位，0位为写命令0）
reg    [7:0]   r_slave_addr_<em>buf</em>; //从设备地址寄存器（地址存高7位，0位为读命令1）
reg    [7:0]   H_byte_addr_<em>buf</em>;    //8位存储器址位
reg    [7:0]   L_byte_addr_<em>buf</em>;    //8位存储器址位
reg    [5:0]   state           ;
reg    [16:0]   cnt             ;    
reg            SCL_r           ;
reg            XCLK_R           ;
reg            sda_out         ;
reg            SDA_en          ;
reg    [3:0]   write_byte_cnt  ;
reg    [7:0]   write_byte_reg  ;
reg            config_done     ;    //完成信号
reg    [31:0]  delay_cnt       ;  
reg    [31:0]  auto_read_write_cnt_reg;  //计数器
reg            wr_flag         ;
reg    [25:0]  wr_flag_cnt     ;
reg            work_en         ; //工作使能信号

assign XCLK = XCLK_R;
assign scl = SCL_r;
assign sda_in = sda;
assign sda = SDA_en ? sda_out : 1&#39;bz;
// assign sda = SDA_en ? <em>(</em>sda_out ? 1&#39;bz : 1&#39;b0<em>)</em> : 1&#39;bz;
assign write_start_flag_wire=<em>(</em>auto_read_write_cnt_reg==CNT_1SEC<em>)</em>?1&#39;b1:1&#39;b0;

// 寄存器表例化
// sccb_ov5640_table u_sccb_ov5640_table<em>(</em>

//     .clk<em>(</em>clk<em>)</em>,

//     .rst_n<em>(</em>rst_n<em>)</em>,
//     .initial_start<em>(</em>initial_start<em>)</em>,
//     .work_done<em>(</em>work_done<em>)</em>,
//     .work_start<em>(</em>work_start<em>)</em>,
//     .initial_data<em>(</em>initial_data<em>)</em>,
//     .initial_done<em>(</em>initial_done<em>)</em>
// <em>)</em>;

//数据复位、开始工作时寄存数据（避免传输中途数据不稳定）
always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
    if <em>(</em>!rst_n<em>)</em> <em>begin</em>
        w_slave_addr_<em>buf</em> &lt;= 8&#39;b0000_0000;//0位为写命令0
        r_slave_addr_<em>buf</em> &lt;= 8&#39;b0000_0001;//0位为读命令1
        H_byte_addr_<em>buf</em>    &lt;= 8&#39;b0;
        L_byte_addr_<em>buf</em>    &lt;= 8&#39;b0;
        data_tx       &lt;= 8&#39;b0;
    <em>end</em> 
    else if <em>(</em>wr_flag<em>)</em> <em>begin</em>
        w_slave_addr_<em>buf</em> [7:1] &lt;= slave_addr; //地址存高7位
        r_slave_addr_<em>buf</em> [7:1] &lt;= slave_addr; //地址存高7位
        data_tx            &lt;=   8&#39;b0011_0001;
        H_byte_addr_<em>buf</em>    &lt;=   8&#39;b0100_0011;//存储器地址
        L_byte_addr_<em>buf</em>    &lt;=   8&#39;b0000_0000;//存储器地址
        // data_tx          &lt;= initial_data[7:0];
        // H_byte_addr_<em>buf</em>    &lt;= initial_data[23:16];//存储器地址
        // L_byte_addr_<em>buf</em>    &lt;= initial_data[15:8];//存储器地址
    <em>end</em>
<em>end</em>

//状态机定时功能
always@<em>(</em>posedge clk or negedge rst_n<em>)</em><em>begin</em>
    if<em>(</em>!rst_n<em>)</em><em>begin</em>
        auto_read_write_cnt_reg&lt;=&#39;d0;
    <em>end</em>
    else <em>begin</em>
        if<em>(</em>auto_read_write_cnt_reg&lt;CNT_3SEC<em>)</em><em>begin</em>
            auto_read_write_cnt_reg&lt;=auto_read_write_cnt_reg+&#39;d1;
        <em>end</em>
        else <em>begin</em>
            auto_read_write_cnt_reg&lt;=&#39;b0;
        <em>end</em>
    <em>end</em>
<em>end</em>

reg    [16:0] cnt_xclk;
always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
    if<em>(</em>!rst_n<em>)</em>
        cnt_xclk &lt;= 0;
    else
    <em>begin</em> 
        if<em>(</em>cnt_xclk == 16&#39;d3<em>)</em>//4000 / 20
            cnt_xclk &lt;= 0;
        else 
            cnt_xclk &lt;= cnt_xclk + 1&#39;b1;
    <em>end</em>
<em>end</em>

always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
    if<em>(</em>!rst_n<em>)</em>
        XCLK_R &lt;= 0;
    else
    <em>begin</em> 
        if<em>(</em>cnt_xclk == 16&#39;d0<em>)</em>
            XCLK_R &lt;= 1&#39;b1;
        else if<em>(</em>cnt == 16&#39;d1<em>)</em>
            XCLK_R &lt;= 1&#39;b0;
    <em>end</em>
<em>end</em>


always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
    if<em>(</em>!rst_n<em>)</em>
        cnt &lt;= 0;
    else
    <em>begin</em> 
        if<em>(</em>cnt == 16&#39;d399<em>)</em>//4000 / 20
            cnt &lt;= 0;
        else 
            cnt &lt;= cnt + 1&#39;b1;
    <em>end</em>
<em>end</em>



always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
    if<em>(</em>!rst_n<em>)</em>
        SCL_r &lt;= 0;
    else
    <em>begin</em> 
        if<em>(</em>cnt == 16&#39;d0<em>)</em>
            SCL_r &lt;= 1&#39;b1;
        else if<em>(</em>cnt == 16&#39;d160<em>)</em>
            SCL_r &lt;= 1&#39;b0;
    <em>end</em>
<em>end</em>


//定义的状态机状态
parameter                           IDLE 				= 6&#39;d0    ;
parameter                           STRAT_W 			= 6&#39;d1    ;
parameter                           WRITE_SLAVE_ADDR 	= 6&#39;d2    ;
parameter                           ACK_1 	            = 6&#39;d3    ;
parameter                           S<em>END</em>_CTRL_BYTE_M 	= 6&#39;d4    ;
parameter                           ACK_2 	            = 6&#39;d5    ;
parameter                           S<em>END</em>_CTRL_BYTE_L 	= 6&#39;d6    ;
// parameter                           ACK_3 	            = 6&#39;d7    ;
// parameter                           S<em>END</em>_DATA        	= 6&#39;d8    ;
parameter                           ACK_4 	            = 6&#39;d7    ;
parameter                           STOP_W	            = 6&#39;d8    ;
// parameter							STRAT_R1			= 6&#39;d11	  ;
// parameter                           S<em>END</em>_CTRL_BYTE_R1 	= 6&#39;d12   ;
// parameter                           ACK_5 	            = 6&#39;d13   ;
// parameter 							CONTROL_BYTE_ADDR_M	= 6&#39;d14   ;
// parameter                           ACK_6 	            = 6&#39;d15   ;
// parameter 							CONTROL_BYTE_ADDR_L	= 6&#39;d16   ;
// parameter                           ACK_7 	            = 6&#39;d17   ;
// parameter                           STOP_R1 			= 6&#39;d18   ;
parameter							STRAT_R2			= 6&#39;d9	  ;
parameter                           S<em>END</em>_CTRL_BYTE_R2 	= 6&#39;d10   ;
parameter                           ACK_8 	            = 6&#39;d11   ;
parameter                           RECV_DATA        	= 6&#39;d12   ;
parameter 							NACK			    = 6&#39;d13   ;
parameter 							WAIT			    = 6&#39;d14   ;
parameter                           STOP_R2 			= 6&#39;d15   ;


//开始信号wr_flag
always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
    if<em>(</em>!rst_n<em>)</em> <em>begin</em> 
        wr_flag_cnt &lt;= 0;
        wr_flag &lt;= 0;
    <em>end</em>
    else <em>begin</em>
        if<em>(</em>wr_flag_cnt &lt; 26&#39;d10_000_000<em>)</em> <em>begin</em> // 100ms延迟
            wr_flag_cnt &lt;= wr_flag_cnt + 1&#39;b1;
            wr_flag &lt;= 1&#39;b0;
        <em>end</em>
        else <em>begin</em> 
            wr_flag_cnt &lt;= 26&#39;d5_000_000; // 保持计数值
            wr_flag &lt;= 1&#39;b1; // 持续触发，或仅触发一次
        <em>end</em>
    <em>end</em>
<em>end</em>
// always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
//  if<em>(</em>!rst_n<em>)</em> <em>begin</em> 
// 		wr_flag_cnt &lt;= 0;
// 		wr_flag &lt;= 0;
//     <em>end</em>
// 	else <em>begin</em>
// 		if<em>(</em>wr_flag_cnt[25]<em>)</em> <em>begin</em> 
// 			wr_flag_cnt &lt;= 0;
// 			wr_flag &lt;= 1&#39;b1;
// 		<em>end</em>
//         else if<em>(</em>wr_flag_cnt&lt;=&#39;d167796<em>)</em> <em>begin</em> 
//             wr_flag_cnt &lt;= wr_flag_cnt + 1&#39;b1;
//             wr_flag &lt;= wr_flag;
//         <em>end</em>
// 		else <em>begin</em> 
// 			wr_flag_cnt &lt;= wr_flag_cnt + 1&#39;b1;
// 			wr_flag &lt;= 1&#39;b0;
// 		<em>end</em>
// 	<em>end</em>
// <em>end</em>

always @<em>(</em>posedge clk or negedge rst_n<em>)</em> <em>begin</em>
    if<em>(</em>!rst_n<em>)</em> <em>begin</em> 
        state &lt;= IDLE;
        write_byte_cnt &lt;= 0;
        write_byte_reg &lt;= 0;
        SDA_en &lt;= 1&#39;b0;
        sda_out &lt;= 1&#39;b1;
        work_en &lt;= 1&#39;b0;
        config_done &lt;= 1&#39;b0;
        //wr_reg &lt;= 0;
    <em>end</em>
    // else if<em>(</em>wr_flag<em>)</em> <em>begin</em> 
    //     state &lt;= IDLE;
    //     write_byte_cnt &lt;= 0;
    //     write_byte_reg &lt;= 0;
    //     SDA_en &lt;= 1&#39;b0;
    //     config_done &lt;= 1&#39;b0;
    // <em>end</em>
    else <em>begin</em> 
        case<em>(</em>state<em>)</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>空闲<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
            IDLE: <em>begin</em> //0
                config_done &lt;= 1&#39;b0;
                SDA_en &lt;= 1&#39;b0;
                sda_out &lt;= 1&#39;b1;
                if<em>(</em>wr_flag<em>)</em><em>begin</em>
                    work_en &lt;= 1&#39;b1;
                    state &lt;= STRAT_W;
                <em>end</em>
                // else 
                //     state &lt;= IDLE;
            <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>写起始位<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
            STRAT_W : <em>begin</em> 
                SDA_en &lt;= 1&#39;b1; //拉高sda
                if<em>(</em>`SCL_HIG_MID<em>)</em> <em>begin</em> 
                    sda_out &lt;= 0;
                    write_byte_cnt &lt;= 0;
                    state &lt;= WRITE_SLAVE_ADDR;
                <em>end</em>
                // else <em>begin</em>
                //     state &lt;= STRAT_W;
                // <em>end</em>
            <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>写器件地址<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
            WRITE_SLAVE_ADDR: <em>begin</em> 
                SDA_en &lt;= 1&#39;b1;
                if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
                    if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em> <em>begin</em>
                        sda_out &lt;= w_slave_addr_<em>buf</em>[7<em>-</em>write_byte_cnt];//sda输出设备地址
                        write_byte_cnt &lt;= write_byte_cnt + 4&#39;d1;
                    <em>end</em> else <em>begin</em>
                        write_byte_cnt &lt;= 4&#39;d0;
                        state &lt;= ACK_1;
                    <em>end</em>
                <em>end</em>
            <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答1<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
            ACK_1: <em>begin</em> 
                SDA_en &lt;= 1&#39;b0;
                if<em>(</em>`SCL_NEGEDGE<em>)</em> 
                    state &lt;= S<em>END</em>_CTRL_BYTE_M;
                // else
                //     state &lt;= ACK_1;                   
                <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>存储器地址高8位<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
            S<em>END</em>_CTRL_BYTE_M: <em>begin</em> 
                SDA_en &lt;= 1&#39;b1;
                if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
                    if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em><em>begin</em>
                        sda_out &lt;= H_byte_addr_<em>buf</em>[7<em>-</em>write_byte_cnt];//sda输出字节地址（从高到低）
                        write_byte_cnt &lt;= write_byte_cnt + 4&#39;d1;
                    <em>end</em> else <em>begin</em>
                        write_byte_cnt &lt;= 4&#39;d0;
                        state &lt;= ACK_2;
                    <em>end</em>
                <em>end</em>
            <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答2<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
                ACK_2: <em>begin</em> 
                    SDA_en &lt;= 1&#39;b0;
                	if<em>(</em>`SCL_NEGEDGE<em>)</em> <em>begin</em>                 
                        state &lt;= S<em>END</em>_CTRL_BYTE_L;
                    <em>end</em>
                	// else
                    // 	state &lt;= ACK_2;                   
            	<em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>存储器地址低8位<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
            S<em>END</em>_CTRL_BYTE_L: <em>begin</em> 
                SDA_en &lt;= 1&#39;b1;
                if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
                    if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em><em>begin</em>
                        sda_out &lt;= L_byte_addr_<em>buf</em>[7<em>-</em>write_byte_cnt];//sda输出字节地址（从高到低）
                        write_byte_cnt &lt;= write_byte_cnt + 1&#39;b1;
                    <em>end</em> else <em>begin</em>
                        write_byte_cnt &lt;= 4&#39;d0;
                        state &lt;= ACK_4;
                    <em>end</em>
                <em>end</em>
            <em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答3<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
//             	ACK_3: <em>begin</em> 
//                     SDA_en &lt;= 1&#39;b0;
//                 	if<em>(</em>`SCL_NEGEDGE<em>)</em> <em>begin</em>                 
//                             state &lt;= S<em>END</em>_DATA;
//                     <em>end</em>
//                 	else
//                     	state &lt;= ACK_3;                   
//             	<em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>写数据<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
//             S<em>END</em>_DATA: <em>begin</em>
//                 SDA_en &lt;= 1&#39;b1;
//                 if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
//                     if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em><em>begin</em>
//                         sda_out &lt;= data_tx[7<em>-</em>write_byte_cnt];//sda输出字节地址（从高到低）
//                         write_byte_cnt &lt;= write_byte_cnt + 1&#39;b1;
//                     <em>end</em> else <em>begin</em>
//                         state &lt;= ACK_4;
//                         write_byte_cnt &lt;= 4&#39;d0;
//                     <em>end</em> 
//                     <em>end</em>    
//                 <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答4<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
                ACK_4: <em>begin</em> 
                    SDA_en &lt;= 1&#39;b0;
                	if<em>(</em>`SCL_NEGEDGE<em>)</em> <em>begin</em>
                	    state &lt;= STOP_W;
                    <em>end</em>
                	else
                    	state &lt;= ACK_4;
            	<em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>写停止<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
                STOP_W: <em>begin</em> 
                    SDA_en &lt;= 1&#39;b1;
                        // sda_out &lt;= 1&#39;b1;
                        // if<em>(</em>delay_cnt ==32&#39;d250_000<em>)</em> <em>begin</em>
                        //     state &lt;= STRAT_R1; 
                        //     delay_cnt &lt;= 0; 
                        // <em>end</em>
                        if<em>(</em>`SCL_HIG_MID<em>)</em> <em>begin</em>
                            sda_out &lt;= 1&#39;b1;
                            state &lt;= STRAT_R2; 
                        <em>end</em>
                        // else
                        //     delay_cnt &lt;= delay_cnt + 1&#39;b1;
                <em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读开始<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
//                 STRAT_R1 : <em>begin</em> 
//                     SDA_en &lt;= 1&#39;b1;
//                 if<em>(</em>`SCL_HIG_MID<em>)</em> <em>begin</em> 
//                     sda_out &lt;= 0;
//                     state &lt;= S<em>END</em>_CTRL_BYTE_R1;
//                 <em>end</em>
//                 else 
//                     state &lt;= STRAT_R1;
// 				<em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读器件地址<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
//                 S<em>END</em>_CTRL_BYTE_R1: <em>begin</em> 
//                     SDA_en &lt;= 1&#39;b1;
//                 if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
//                     if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em> <em>begin</em>
//                         sda_out &lt;= w_slave_addr_<em>buf</em>[7<em>-</em>write_byte_cnt];//sda输出设备地址
//                         write_byte_cnt &lt;= write_byte_cnt + 1&#39;b1;
//                     <em>end</em> else <em>begin</em>
//                         state &lt;= ACK_5;
//                         write_byte_cnt &lt;= 4&#39;d0;
//                     <em>end</em> 
//                     <em>end</em>
//                 <em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答5<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
//                 ACK_5: <em>begin</em> 
//                     SDA_en &lt;= 1&#39;b0;
//                     if<em>(</em>`SCL_NEGEDGE<em>)</em> <em>begin</em>
//                         state &lt;= CONTROL_BYTE_ADDR_M;
//                     <em>end</em>
//                     else
//                     state &lt;= ACK_5;
//                 <em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读存储器高8<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
// 				CONTROL_BYTE_ADDR_M: <em>begin</em>
//                     SDA_en &lt;= 1&#39;b1;
// 				if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
//                     if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em><em>begin</em>
//                         sda_out &lt;= H_byte_addr_<em>buf</em>[7<em>-</em>write_byte_cnt];//sda输出字节地址（从高到低）
//                         write_byte_cnt &lt;= write_byte_cnt + 1&#39;b1;
//                     <em>end</em> else <em>begin</em>
//                         state &lt;= ACK_6;
//                         write_byte_cnt &lt;= 4&#39;d0;
//                     <em>end</em>
//                 <em>end</em>
// 				<em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答6<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
//                 ACK_6: <em>begin</em> 
//                     SDA_en &lt;= 1&#39;b0;
//                     if<em>(</em>`SCL_NEGEDGE<em>)</em> <em>begin</em>
//                         state &lt;= CONTROL_BYTE_ADDR_L;
//                     <em>end</em>
//                     else
//                     state &lt;= ACK_6;
//                 <em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读存储器低8<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
// 				CONTROL_BYTE_ADDR_L: <em>begin</em>
//                     SDA_en &lt;= 1&#39;b1;
// 				if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
//                     if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em><em>begin</em>
//                         sda_out &lt;= L_byte_addr_<em>buf</em>[7<em>-</em>write_byte_cnt];//sda输出字节地址（从高到低）
//                         write_byte_cnt &lt;= write_byte_cnt + 1&#39;b1;
//                     <em>end</em> else <em>begin</em>
//                         state &lt;= ACK_7;
//                         write_byte_cnt &lt;= 4&#39;d0;
//                     <em>end</em>
//                 <em>end</em>
// 				<em>end</em>             
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答7<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
// 				ACK_7: <em>begin</em> 
//                     SDA_en &lt;= 1&#39;b0;
//                     if<em>(</em>`SCL_NEGEDGE<em>)</em> <em>begin</em>
//                         state &lt;= STOP_R1;
//                     <em>end</em>
//                     else
//                     state &lt;= ACK_7;
//                 <em>end</em>
// //<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读停止1<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
//                 STOP_R1: <em>begin</em> //7
//                     SDA_en &lt;= 1&#39;b1;
//                     if<em>(</em>`SCL_HIG_MID<em>)</em> <em>begin</em>
//                         sda_out &lt;= 1&#39;b1;
//                         state &lt;= STRAT_R2; 
//                     <em>end</em>
//                     else
//                     state &lt;= STOP_R1;
//                 <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读开始2<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
                STRAT_R2 : <em>begin</em> //1
                SDA_en &lt;= 1&#39;b1;
                if<em>(</em>`SCL_HIG_MID<em>)</em> <em>begin</em> 
                    sda_out &lt;= 1&#39;b0;
                    state &lt;= S<em>END</em>_CTRL_BYTE_R2;
                <em>end</em>
                // else 
                //     state &lt;= STRAT_R2;
				<em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读器件地址<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
                S<em>END</em>_CTRL_BYTE_R2: <em>begin</em> 
                    SDA_en &lt;= 1&#39;b1;
                if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
                    if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em> <em>begin</em>
                        sda_out &lt;= r_slave_addr_<em>buf</em>[7<em>-</em>write_byte_cnt];//sda输出设备地址
                        write_byte_cnt &lt;= write_byte_cnt + 4&#39;d1;
                    <em>end</em> else <em>begin</em>
                        write_byte_cnt &lt;= 4&#39;d0;
                        state &lt;= ACK_8;
                    <em>end</em> 
                    <em>end</em>
                <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>应答8<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
                ACK_8: <em>begin</em> 
                    SDA_en &lt;= 1&#39;b0;
                    if<em>(</em>`SCL_NEGEDGE<em>)</em> <em>begin</em>
                        state &lt;= RECV_DATA;
                    <em>end</em>
                    // else
                    // state &lt;= ACK_8;
                <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>接收数据<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
                RECV_DATA: <em>begin</em>
                    SDA_en &lt;= 1&#39;b0; 
                    if<em>(</em>`SCL_HIG_MID<em>)</em><em>begin</em>
                    if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em><em>begin</em>
                        data_rx[7<em>-</em>write_byte_cnt] &lt;= sda_in;
                        write_byte_cnt &lt;= write_byte_cnt + 4&#39;d1;
                    <em>end</em>
                    if<em>(</em>`SCL_HIG_MID &amp;&amp; write_byte_cnt == 4&#39;d8<em>)</em> <em>begin</em>
                        state &lt;= NACK;
                        write_byte_cnt &lt;= 4&#39;d0;
                    <em>end</em>
                <em>end</em>
                <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>无应答<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
			NACK: <em>begin</em> 
                SDA_en &lt;= 1&#39;b1;
                if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
                        sda_out &lt;= 1&#39;b1;
                        state &lt;= WAIT;
                <em>end</em>
                // else
                //     state &lt;= NACK;
            <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>等待<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
            WAIT: <em>begin</em>
                SDA_en &lt;= 1&#39;b0;
                if<em>(</em>`SCL_LOW_MID<em>)</em> <em>begin</em>
                    sda_out &lt;= 1&#39;b0;
                    state &lt;= STOP_R2;
                <em>end</em>
                else
                    state &lt;= WAIT;
            <em>end</em>
//<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>读结束<em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em><em>-</em>//
			STOP_R2: <em>begin</em> 
                SDA_en &lt;= 1&#39;b1;
                 if<em>(</em>`SCL_HIG_MID<em>)</em> <em>begin</em>
                    config_done &lt;= 1&#39;b1;
                    sda_out &lt;= 1&#39;b1;
                    state &lt;= IDLE;
                <em>end</em>
            <em>end</em>
            
<em>end</em>case
<em>end</em>

<em>end</em>



ila_0 ila_0_debuge <em>(</em>
	.clk<em>(</em>clk<em>)</em>, // input wire clk


	.probe0<em>(</em>{clk,rst_n,sda,SCL_r,state,wr_flag_cnt,wr_flag,config_done,write_byte_cnt,cnt,sda_in,sda_out,SDA_en,data_tx,data_rx,delay_cnt,work_en}<em>)</em> // input wire [31:0] probe0
<em>)</em>;


<em>end</em>module  检查一下这段代码，为什么无法接受来自摄像头里寄存器的数据？</div>
					<div class="tag">最新发布</div>
				</a>
			</div>
			<div class="info-box display-flex">
				<div class="info display-flex">
					<span class="info-block">08-28</span>
				</div>
			</div>
		</div>
		<div class="desc-box">
			<a href="https://wenku.csdn.net/answer/6m8pdk8q94" target="_blank"  data-report-click='{"ab":"new","spm":"1001.2101.3001.6650.19","mod":"popu_387","extra":"{\"highlightScore\":0.0,\"utm_medium\":\"distribute.pc_relevant.none-task-chatgpt-2~default~OPENSEARCH~Position-19-6m8pdk8q94-blog-102596825.235^v43^pc_blog_bottom_relevance_base9\",\"dist_request_id\":\"1766382508835_16991\"}","dist_request_id":"1766382508835_16991","ab_strategy":"landing_bge_commercial_hanlp_download_sign","index":"19","strategy":"2~default~OPENSEARCH~Position","dest":"https://wenku.csdn.net/answer/6m8pdk8q94"}'  data-report-query='spm=1001.2101.3001.6650.19&utm_medium=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-chatgpt-2%7Edefault%7EOPENSEARCH%7EPosition-19-6m8pdk8q94-blog-102596825.235%5Ev43%5Epc_blog_bottom_relevance_base9'>
				<div class="desc ellipsis-online ellipsis-online-1">这段代码是一个 SCCB（Serial Camera Control Bus）控制器的 Verilog 实现，用于与摄像头（如 OV5640）进行通信。你提到的问题是&ldquo;无法接受来自摄像头里寄存器的数据&rdquo;，我们可以从以下几个方面来分析问题：

<em>-</em><em>-</em><em>-</em>

### 1. **读取流程是否完整**
SCCB 协议中，读取寄存器的流程是：
<em>-</em> 发送 Start 信号
<em>-</em> 发送写地址（写模式）
<em>-</em> 发送寄存器地址（高 + 低字节）
<em>-</em> 发送 Start 信号（重复起始）
<em>-</em> 发送读地址（读模式）
<em>-</em> 接收数据
<em>-</em> 发送 Nack
<em>-</em> 发送 Stop 信号

你当前的代码中：
<em>-</em> 有写地址和寄存器地址的发送流程（`STRAT_W`, `WRITE_SLAVE_ADDR`, `S<em>END</em>_CTRL_BYTE_M`, `S<em>END</em>_CTRL_BYTE_L`）
<em>-</em> 也有读地址发送流程（`STRAT_R2`, `S<em>END</em>_CTRL_BYTE_R2`, `ACK_8`, `RECV_DATA`）
<em>-</em> 但 Nack 和 Stop 的处理（`NACK`, `WAIT`, `STOP_R2`）逻辑可能存在问题

**问题点：**
<em>-</em> 在 `RECV_DATA` 状态中，接收完 8 位数据后进入 `NACK` 状态，但 Nack 的处理是否正确？
<em>-</em> `NACK` 状态中是否拉高 SDA（发送 Nack）？
<em>-</em> `WAIT` 状态是否有必要？是否可能导致状态机卡住？
<em>-</em> `STOP_R2` 是否正确发送 Stop 信号并回到 `IDLE`？

<em>-</em><em>-</em><em>-</em>

### 2. **SDA 输入是否正确采样**
在 `RECV_DATA` 状态中：
```verilog
RECV_DATA: <em>begin</em>
    SDA_en &lt;= 1&#39;b0; 
    if<em>(</em>`SCL_HIG_MID<em>)</em><em>begin</em>
        if<em>(</em>write_byte_cnt != 4&#39;d8<em>)</em><em>begin</em>
            data_rx[7<em>-</em>write_byte_cnt] &lt;= sda_in;
            write_byte_cnt &lt;= write_byte_cnt + 4&#39;d1;
        <em>end</em>
        if<em>(</em>`SCL_HIG_MID &amp;&amp; write_byte_cnt == 4&#39;d8<em>)</em> <em>begin</em>
            state &lt;= NACK;
            write_byte_cnt &lt;= 4&#39;d0;
        <em>end</em>
    <em>end</em>
<em>end</em>
```

**问题点：**
<em>-</em> `sda_in` 是否正确采样？应确保在 SCL 高电平中间采样（即 `SCL_HIG_MID`），但当前逻辑可能存在问题：
  <em>-</em> 多重 `if<em>(</em>`SCL_HIG_MID<em>)</em>` 判断可能导致采样时机不准确
  <em>-</em> `write_byte_cnt` 是否在 SCL 高电平中间递增？

<em>-</em><em>-</em><em>-</em>

### 3. **主时钟与 SCCB 时钟匹配问题**
SCCB 的 SCL 频率通常为 100kHz 或 400kHz，而你的主时钟为 100MHz。

```verilog
`define SCL_POSEDGE <em>(</em>cnt == 16&#39;d0<em>)</em>
`define SCL_NEGEDGE <em>(</em>cnt == 16&#39;d160<em>)</em>
`define SCL_HIG_MID <em>(</em>cnt == 16&#39;d80<em>)</em>
`define SCL_LOW_MID <em>(</em>cnt == 16&#39;d280<em>)</em>
```

**问题点：**
<em>-</em> `cnt` 的最大值为 399，一个完整的 SCL 周期为 400 个主时钟周期（即 100MHz / 400 = 250kHz），这已经高于标准 SCCB 的最高频率（400kHz），是否摄像头支持这么高的频率？
<em>-</em> 如果摄像头不支持高频，可能导致通信失败

<em>-</em><em>-</em><em>-</em>

### 4. **I2C 和 SCCB 的差异**
SCCB 是 I2C 的子集，但 SCCB 不支持连续读写、时序更严格。

**问题点：**
<em>-</em> 是否摄像头要求某些固定延时或特定时序？
<em>-</em> 是否在发送完寄存器地址后有足够延时才开始读操作？
<em>-</em> 是否遗漏了某些必要的延时（如 `delay_cnt` 被注释掉了）？

<em>-</em><em>-</em><em>-</em>

### 5. **调试建议**
1. 使用 ILA（你已经例化了 `ila_0`）观察以下信号：
   <em>-</em> `sda`, `scl`, `state`, `write_byte_cnt`, `data_rx`
   <em>-</em> 检查是否成功进入 `RECV_DATA` 状态
   <em>-</em> 检查 `data_rx` 是否有数据变化
2. 使用示波器/逻辑分析仪抓取 SCCB 信号，确认 SDA 和 SCL 波形是否符合规范
3. 确保摄像头已正确上电并处于正常工作模式（`cam_pwdn`, `cam_rst` 是否正确控制）

<em>-</em><em>-</em><em>-</em>

### 6. **建议修改点**
<em>-</em> 优化 `RECV_DATA` 状态的采样逻辑，确保在 SCL 高电平中间采样
<em>-</em> 确保 `NACK` 正确发出（SDA 拉高）
<em>-</em> 在 `STOP_R2` 中确保 Stop 信号正确发出
<em>-</em> 降低 SCCB 时钟频率（如 100kHz）以确保兼容性

<em>-</em><em>-</em><em>-</em>

### 示例修正代码（RECV_DATA 状态）：
```verilog
RECV_DATA: <em>begin</em>
    SDA_en &lt;= 1&#39;b0; // 输入模式
    if<em>(</em>`SCL_HIG_MID<em>)</em> <em>begin</em>
        if<em>(</em>write_byte_cnt &lt; 8<em>)</em> <em>begin</em>
            data_rx[7 <em>-</em> write_byte_cnt] &lt;= sda_in; // 采样 SDA
            write_byte_cnt &lt;= write_byte_cnt + 1;
        <em>end</em> else <em>begin</em>
            state &lt;= NACK;
            write_byte_cnt &lt;= 0;
        <em>end</em>
    <em>end</em>
<em>end</em>
```

<em>-</em><em>-</em><em>-</em>

###</div>
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
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（五） —— File
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596845" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（七） —— alloc page 版本
            </a>
          </dd>
      </dl>
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
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596802" data-report-query="spm=3001.10752">
              dma-buf 由浅入深（五） —— File
            </a>
          </dd>
      </dl>
      <dl class="next" data-report-click='{"spm":"3001.10796","extend1":"下一篇"}' data-report-view='{"spm":"3001.10796","extend1":"下一篇"}'>
          <dt>
            下一篇：
          </dt>
          <dd >
            <a href="https://blog.csdn.net/hexiaolong2009/article/details/102596845" data-report-query="spm=3001.10796">
              dma-buf 由浅入深（七） —— alloc page 版本
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
		<div class="comment-side-tit-count">评论&nbsp;<span class="count">3</span></div>
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
              <input type="hidden" id="article_id" name="article_id" value="102596825">
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
