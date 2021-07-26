class AmazonProductScrapper < Kimurai::Base

  def self.scrape(product_url)
    product_url = product_url.split("dp/").first + "product-reviews/" + product_url.split("dp/").last[0..-1]
    @start_urls = [product_url]
    # self.crawl!
    return self.parse!(:parse, url: product_url)
  end
  @name = "amazon_product_scrapper"
  # @engine = :selenium_chrome

  def parse(response, url:, data: {})
    @reviews_results ||= []
    page_url = url
    page_url = request_to :reviews_scrape, url: page_url
    while(page_url.present?)
      puts "----> #{page_url}"
      page_url = URI.encode("https://www.amazon.com#{page_url.attribute("href").value.squish}")
      page_url = request_to :reviews_scrape, url: page_url
    end
    return @reviews_results
  end

  def reviews_scrape(response, url:, data: )
    response.css(".a-section.review.aok-relative").each do |a|
      next if a.css(".a-section.a-spacing-none.a-spacing-top-small.cr-translate-this-review-section").present?
      review = a.css(".a-size-base.review-text.review-text-content span:not(.aok-hidden)").text.squish
      @reviews_results << review if review.present?
    end

    next_page = response.css(".a-pagination li:last-of-type a")
    return next_page
  end

end